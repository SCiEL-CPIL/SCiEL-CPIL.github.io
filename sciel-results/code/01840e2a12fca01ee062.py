import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = 256

    def check_valid(seq):
        """Return True if all non-empty consecutive block sums are distinct."""
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def greedy_insert(seq, candidates):
        """
        Greedily insert as many candidates as possible into seq.
        Candidates are tried in the given order.
        Returns (new_seq, count_inserted).
        """
        new_seq = list(seq)
        count = 0
        for x in candidates:
            if x in new_seq:
                continue
            # Try to find a valid position to insert x
            # Since seq is sorted, we can find the position where x would go
            # and check validity around that area. But for correctness, 
            # we just try all positions? No, that's O(n) checks per insert, 
            # and check_valid is O(n^2). 
            # Better: binary search for position, then verify.
            # Actually, since it's strictly increasing, x goes in a specific spot.
            # If x is already in new_seq, skip.
            # Find insertion point
            lo, hi = 0, len(new_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if new_seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            # Check if x is already present (should be handled by 'if x in new_seq')
            if pos < len(new_seq) and new_seq[pos] == x:
                continue
            
            # Create new sequence
            candidate_seq = new_seq[:pos] + [x] + new_seq[pos:]
            
            # Optimization: Instead of full check_valid, we can be smarter?
            # For now, full check is safe. If too slow, we'll optimize.
            if check_valid(candidate_seq):
                new_seq = candidate_seq
                count += 1
        return new_seq, count

    def greedy_init(order):
        """Greedy: add elements in given order if valid."""
        seq = []
        for x in order:
            new_seq = sorted(seq + [x])
            if check_valid(new_seq):
                seq = new_seq
        return seq

    # Phase 1: Generate a strong initial sequence using multiple greedy strategies
    best_seq = []
    best_len = 0

    # Standard ascending
    seq = greedy_init(list(range(1, N + 1)))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # Descending
    seq = greedy_init(list(range(N, 0, -1)))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # Random shuffles
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        seq = greedy_init(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Phase 2: Local Search via Adjacent Swaps (2-opt style)
    # Objective: Maximize the number of elements we can add to the current sequence.
    # State: A valid sequence S.
    # Move: Swap two adjacent elements in S? 
    # Wait, S is sorted. Swapping adjacent elements in a sorted sequence breaks the sorted order.
    # The problem requires a "strictly increasing sequence". 
    # So the sequence MUST be sorted. 
    # Therefore, we cannot swap elements within the sequence to change the order.
    # The "structure" is the set of elements.
    
    # Revised Strategy:
    # The "landscape" is defined by the set of elements.
    # My previous idea of "adjacent swaps" in the sequence doesn't work because the sequence is always sorted.
    # However, we can define moves on the SET.
    # Let's stick to the set-based local search but with a smarter neighbor generation.
    
    # Alternative Structural Change:
    # Instead of random add/remove/swap, use a **Hill Climbing with Restart** on the set, 
    # but define the "energy" not just as length, but as **potential**.
    # Or, better: **Genetic Algorithm** approach?
    # Let's try a different angle: **Constraint Programming / Backtracking with Heuristics**.
    # The current program uses SA. Let's try a **Greedy with Lookahead** that is more systematic.
    
    # Actually, let's look at the "Inspiration 3" which mentioned building backwards or different framing.
    # Let's try a **Backtracking Search** that tries to build a sequence of length L.
    # We binary search on L. For a given L, we try to find a valid sequence of length L.
    # This is a different optimization paradigm: Feasibility search vs Optimization.
    
    def find_sequence_of_length(L, time_limit):
        """
        Try to find a valid sequence of exactly length L.
        Uses backtracking with heuristics.
        """
        if L <= 0:
            return []
        if L > N:
            return None
            
        # Heuristic: Start with a greedy sequence of length L?
        # Or start with the smallest L numbers?
        # Let's use a randomized greedy to get a starting set, then backtrack.
        
        # Actually, backtracking on 256 choose L is huge.
        # Let's stick to the local search but change the move generation.
        
        pass

    # Let's implement a **Tabu Search** or **Iterated Local Search (ILS)**.
    # ILS: 
    # 1. Local Search to convergence (Hill Climbing).
    # 2. Perturbation (random restarts or large random changes).
    # 3. Local Search again.
    
    def hill_climb(seq, time_limit):
        """
        Hill Climbing on the set of elements.
        Moves: Add one, Remove one, Swap one.
        Accept if length increases. (Strict improvement only, to escape plateaus we use ILS).
        """
        current_set = set(seq)
        current_seq = sorted(current_set)
        current_len = len(current_seq)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all possible single additions
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                new_set = current_set | {x}
                new_seq = sorted(new_set)
                if check_valid(new_seq):
                    current_set = new_set
                    current_seq = new_seq
                    current_len += 1
                    improved = True
                    # Restart loop to check again from new state
                    break
            # If no single addition works, try swaps?
            # Hill climbing usually stops at local optima. 
            # To escape, we need to allow worse moves or perturbations.
            # For pure Hill Climbing, we stop here.
            # Let's add a "swap" phase if no additions are possible.
            if not improved:
                # Try swaps to see if we can enable further additions
                swaps_found = False
                for i, x_out in enumerate(current_seq):
                    for x_in in range(1, N + 1):
                        if x_in in current_set:
                            continue
                        new_set = current_set - {x_out} | {x_in}
                        new_seq = sorted(new_set)
                        if check_valid(new_seq):
                            # Check if this swap allows an addition
                            # We accept the swap only if it leads to a longer sequence 
                            # OR if it's the same length but we are stuck? 
                            # Standard HC: only accept if better. Same length is not better.
                            # But we are stuck. 
                            # Let's accept same-length swaps if they increase "potential"?
                            # For simplicity, let's just do ILS: Perturb then HC.
                            pass
                # If we are here, we are at a local optimum for this HC.
        
        return current_seq

    # Let's implement **Iterated Local Search (ILS)** properly.
    # 1. Get a local optimum via Hill Climbing (Add-only HC).
    # 2. Perturb: Remove k random elements.
    # 3. Hill Climb again.
    
    def ilds(seq, time_limit):
        """Iterated Local Search."""
        current_seq = list(seq)
        best_seq = current_seq[:]
        best_len = len(current_seq)
        
        while time.time() < time_limit:
            # 1. Local Search: Greedy Add
            # This is fast.
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for x in range(1, N + 1):
                    if x in current_seq:
                        continue
                    new_seq = sorted(current_seq + [x])
                    if check_valid(new_seq):
                        current_seq = new_seq
                        changed = True
                        break
            
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
            
            # 2. Perturbation
            # Remove 5-10 random elements
            if not current_seq:
                break
            k = random.randint(3, 10)
            to_remove = random.sample(current_seq, min(k, len(current_seq)))
            current_seq = [x for x in current_seq if x not in set(to_remove)]
            
        return best_seq

    # Run ILS multiple times
    if time.time() < deadline - 5:
        budget = deadline - 5 - time.time()
        # Split budget among several runs
        num_runs = 10
        run_budget = budget / num_runs
        
        for i in range(num_runs):
            if time.time() > deadline - 5:
                break
            
            # Different initializations
            if i == 0:
                init_seq = best_seq
            else:
                # Random perturbation of best_seq or new random greedy
                if random.random() < 0.5 and best_seq:
                    init_seq = list(best_seq)
                    k = random.randint(5, 15)
                    to_remove = random.sample(init_seq, min(k, len(init_seq)))
                    init_seq = [x for x in init_seq if x not in set(to_remove)]
                else:
                    order = list(range(1, N + 1))
                    random.shuffle(order)
                    init_seq = greedy_init(order)
            
            result = ilds(init_seq, min(run_budget, deadline - 5 - time.time()))
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]

    return best_seq
