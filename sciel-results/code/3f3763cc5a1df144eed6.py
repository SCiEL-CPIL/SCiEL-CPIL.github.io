import time
import random
from fractions import Fraction
import heapq

def _get_primes(limit=1000):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_cap = 100000
    primes_set = set(primes)
    recip_map = {p: Fraction(1, p) for p in primes}
    
    # We will implement a Greedy Set Cover approach with backtracking.
    # State:
    # - selected: list of (p, r)
    # - used_primes: set of p
    # - current_budget: Fraction
    # - covered: boolean array or set of covered numbers up to current L
    # - current_L: the prefix length covered
    
    # To make this efficient, we don't maintain a full boolean array of size 100k for every step.
    # Instead, we maintain the set of "gaps" or just check the next number.
    # Actually, checking if a number is covered is O(K) where K is number of primes.
    # If K is ~100, checking 1..L is O(L*K).
    # A better way: maintain a sorted list of covered intervals? No, APs are sparse.
    # Let's stick to a boolean array `is_covered` of size L_cap+1, but update it incrementally.
    
    def calc_L_from_counts(counts, max_check=L_cap):
        L = 0
        for i in range(1, max_check + 1):
            if counts[i] > 0:
                L = i
            else:
                break
        return L

    def greedy_with_backtracking():
        # We'll use a recursive or stack-based search.
        # Given the constraints, a pure DFS might be too slow if branching factor is high.
        # However, the "greedy" aspect restricts branching: we only consider moves that improve L.
        # If no move improves L, we are stuck.
        
        # Let's try a hybrid:
        # 1. Greedy construction to get a good initial L.
        # 2. Local search (SA or Hill Climbing) on the residues of the selected primes to polish.
        
        # Phase 1: Greedy Construction
        # Start with empty set.
        # At each step, find the (p, r) that maximizes new_L, subject to budget.
        # To speed this up, we don't try all p, r.
        # We know that to cover n, we need a prime p such that n % p = r.
        # For a given n, the possible (p, r) pairs are (p, n % p) for all p not used.
        # We want to cover the *first* gap.
        
        counts = [0] * (L_cap + 1)
        selected = [] # list of (p, r)
        used_primes = set()
        current_budget = Fraction(1)
        current_L = 0
        
        # Precompute reciprocals for speed
        # We'll iterate through primes sorted by size? Or random?
        # Let's try to cover 1, 2, 3...
        
        # Optimization: 
        # Instead of scanning all p, r for every step, we can identify the "critical" numbers.
        # The first gap is current_L + 1.
        # To cover current_L + 1, we can pick any unused prime p and set r = (current_L + 1) % p.
        # This guarantees covering current_L + 1.
        # Does it cover more? It covers current_L + 1 + k*p.
        # So the gain depends on how many numbers in the range [current_L+2, ...] are also hit.
        
        # Let's implement a "Best Move" search.
        # Candidates: All unused primes p.
        # For each p, r is fixed to cover the first gap: r = (current_L + 1) % p.
        # Wait, is it always optimal to cover the first gap?
        # Yes, because L is defined by the first gap. Any move that doesn't cover the first gap
        # leaves L unchanged (or decreases it if we remove something, but we're adding).
        # So, to increase L, we MUST cover the current first gap.
        # Therefore, for any unused prime p, the ONLY residue r that can increase L 
        # (by covering the first gap) is r = (current_L + 1) % p.
        # (Unless the first gap is already covered by other primes? No, by definition it's the first gap).
        
        # So the search space for the next move is just the choice of prime p!
        # We evaluate the gain in L for each unused prime p if we add (p, (current_L+1)%p).
        # We pick the p that maximizes the new L, subject to budget.
        
        # This is a very structured greedy algorithm.
        
        max_iterations = 200
        stuck_count = 0
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
                
            first_gap = current_L + 1
            if first_gap > L_cap:
                break
                
            best_new_L = current_L
            best_p = None
            best_r = None
            best_cost = None
            
            # Try all unused primes
            # To make it faster, we can limit to primes that are "small enough" to be useful?
            # Large primes cover fewer numbers in the range.
            # But small primes might be out of budget.
            
            # Sort unused primes? We can just iterate.
            # Number of primes is 168.
            # For each p, we calculate new coverage.
            
            # Optimization: 
            # Instead of rebuilding counts for every p, we can simulate.
            # But calculating new L requires knowing if current_L+2, +3, etc. are covered.
            # current_L+1 will be covered by definition.
            # Let's just do the full check for the top few candidates?
            # Or just do it for all. 168 primes * O(L/p) updates?
            # L is up to 100k. O(L) per prime is 168 * 100k = 16.8M ops.
            # This is too slow for Python if done naively in a loop.
            
            # Better: 
            # We only need to check how far the coverage extends.
            # We know 1..current_L are covered.
            # We add p with r = (current_L+1)%p.
            # New covered numbers: current_L+1, current_L+1+p, current_L+1+2p, ...
            # We need to find the first number n > current_L such that n is NOT covered 
            # by the existing set AND NOT in the new AP.
            
            # Let's maintain a set of "uncovered" numbers? No, too big.
            # Let's maintain `counts` array.
            # To evaluate a candidate p:
            # 1. Tentatively add p to counts.
            # 2. Scan from current_L+1 to find first gap.
            # 3. Remove p from counts.
            # This is O(L) per candidate.
            
            # To speed up, we can stop scanning early if we find a gap.
            
            # Let's try this.
            
            # Pre-filter primes: only those that fit in budget.
            # current_budget - 1/p >= 0 => 1/p <= current_budget
            
            candidates = []
            for p in primes:
                if p in used_primes:
                    continue
                cost = recip_map[p]
                if current_budget - cost < 0:
                    continue
                candidates.append(p)
                
            if not candidates:
                break
                
            # To avoid O(N_candidates * L), we can be smarter.
            # But let's try the naive simulation first with a time check.
            
            # We'll evaluate candidates in random order to break ties?
            # Or sort by p? Smaller p covers more densely.
            # Let's sort by p ascending.
            candidates.sort()
            
            for p in candidates:
                r = (first_gap - 1) % p # Wait, first_gap is 1-indexed. 
                # If we want to cover number N, r = N % p.
                # If N % p == 0, r = 0.
                # The AP is r, r+p, r+2p...
                # If r=0, it covers p, 2p...
                # My previous logic: start = r if r>=1 else r+p.
                # Let's stick to that.
                
                # Calculate new L
                # We need to check coverage from current_L+1 onwards.
                # We know current_L+1 is covered by new p.
                
                # Let's do a quick scan.
                # We'll use the existing `counts` array.
                # Add p
                start = r if r >= 1 else r + p
                for n in range(start, L_cap + 1, p):
                    counts[n] += 1
                    
                # Find new L
                new_L = current_L
                for i in range(first_gap, L_cap + 1):
                    if counts[i] > 0:
                        new_L = i
                    else:
                        break
                        
                # Remove p
                for n in range(start, L_cap + 1, p):
                    counts[n] -= 1
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_p = p
                    best_r = r
                    best_cost = cost
                    
                # Early stop if we reached cap?
                if best_new_L >= L_cap:
                    break
            
            if best_p is None or best_new_L <= current_L:
                # No improvement
                break
                
            # Apply the move
            selected.append((best_p, best_r))
            used_primes.add(best_p)
            current_budget -= best_cost
            
            start = best_r if best_r >= 1 else best_r + best_p
            for n in range(start, L_cap + 1, best_p):
                counts[n] += 1
                
            current_L = best_new_L
            
        return selected, counts, current_L

    # Run the greedy construction
    selected, counts, current_L = greedy_with_backtracking()
    
    best_L = current_L
    best_selected = selected[:]
    
    # Phase 2: Local Search (SA) to polish the residues.
    # The greedy step fixed residues to cover the first gap.
    # But shifting a residue might cover more later numbers, or allow a different prime to be used?
    # No, the prime set is fixed. We just adjust residues.
    
    # We use the same SA as before but starting from this greedy solution.
    # This should be much faster to converge because the starting point is "structured".
    
    class State:
        def __init__(self, selected, counts):
            self.primes = [p for p, r in selected]
            self.residues = [r for p, r in selected]
            self.counts = counts
            self.L_cap = L_cap
            
        def get_L(self):
            L = 0
            for i in range(1, self.L_cap + 1):
                if self.counts[i] > 0:
                    L = i
                else:
                    break
            return L
            
        def apply_change(self, idx, new_r):
            old_p = self.primes[idx]
            old_r = self.residues[idx]
            
            if old_r == new_r:
                return self.get_L()
                
            # Remove old
            start_old = old_r if old_r >= 1 else old_r + old_p
            for n in range(start_old, self.L_cap + 1, old_p):
                self.counts[n] -= 1
                
            # Add new
            start_new = new_r if new_r >= 1 else new_r + old_p
            for n in range(start_new, self.L_cap + 1, old_p):
                self.counts[n] += 1
                
            self.residues[idx] = new_r
            return self.get_L()

    if selected:
        state = State(selected, counts)
        current_L = state.get_L()
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < deadline - 5:
            idx = random.randint(0, len(state.primes) - 1)
            p = state.primes[idx]
            old_r = state.residues[idx]
            
            # Smart move: try to cover the first gap or a gap near it
            gap = state.get_L() + 1
            if gap <= L_cap:
                # Try residue that covers gap
                r_target = gap % p
                if random.random() < 0.7:
                    new_r = r_target
                else:
                    new_r = random.randint(0, p - 1)
            else:
                new_r = random.randint(0, p - 1)
                
            new_L = state.apply_change(idx, new_r)
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_selected = [(state.primes[i], state.residues[i]) for i in range(len(state.primes))]
            else:
                prob = pow(2.71828, delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    state.apply_change(idx, old_r)
                    current_L = state.get_L()
                    
            T *= cooling
            if T < T_min:
                T = T_min

    # Final check
    if not best_selected:
        # Fallback if greedy failed completely (unlikely)
        # Just return a valid small set
        return [[13, 1]]
        
    # Verify budget
    total_sum = sum(Fraction(1, p) for p, r in best_selected)
    if total_sum > Fraction(1):
        # Should not happen, but just in case
        # Remove largest primes
        best_selected.sort(key=lambda x: 1/x[0], reverse=True)
        while total_sum > Fraction(1) and best_selected:
            p, r = best_selected.pop()
            total_sum -= Fraction(1, p)
            
    return [[p, r] for p, r in best_selected]
