import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 88.0 # 2s margin
    
    # Precompute length contributions
    # chord k contributes: k+1, N-k+1
    # pair (u,v) contributes: abs(u-v)+2
    
    def get_covered_lengths(S):
        covered = {N} # Rim
        for k in S:
            covered.add(k + 1)
            covered.add(N - k + 1)
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u = S_list[i]
                v = S_list[j]
                covered.add(abs(u - v) + 2)
        return covered

    def is_pancyclic(S):
        covered = get_covered_lengths(S)
        for L in range(3, N):
            if L not in covered:
                return False
        return True

    def build_result(S):
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords
        for k in S:
            edges.append((0, k))
            
        # Cycles
        cycles = []
        # We need to map each length 3..N to a specific cycle
        # Length N: Rim
        # Length k+1: 0, 1, ..., k
        # Length N-k+1: 0, N-1, ..., k
        # Length |u-v|+2: 0, u, ..., v
        
        length_to_cycle = {}
        
        # Rim
        length_to_cycle[N] = list(range(N))
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_to_cycle:
                length_to_cycle[l1] = [0] + list(range(1, k + 1))
            if l2 not in length_to_cycle:
                length_to_cycle[l2] = [0] + list(range(N - 1, k - 1, -1))
                
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            for j in range(i + 1, len(S_sorted)):
                u = S_sorted[i]
                v = S_sorted[j]
                l = abs(u - v) + 2
                if l not in length_to_cycle:
                    length_to_cycle[l] = [0, u] + list(range(u + 1, v + 1))
                    
        # Construct list in order 3..N
        for L in range(3, N + 1):
            if L in length_to_cycle:
                cycles.append(length_to_cycle[L])
            else:
                return None # Should not happen if is_pancyclic passed
                
        return {'edges': edges, 'cycles': cycles}

    candidates = list(range(2, N - 1))
    best_S = None
    
    # Search for minimal m
    # We know m=6 might be possible, but let's be safe and start lower?
    # Theoretical lower bound: Each chord adds 2 single lengths + pairs.
    # With m chords, we get 2m single lengths and m(m-1)/2 pair lengths.
    # Total lengths covered approx 2m + m^2/2.
    # We need 37 lengths.
    # m=5: 10 + 10 = 20 (too few)
    # m=6: 12 + 15 = 27 (too few? maybe overlaps reduce count)
    # m=7: 14 + 21 = 35 (close)
    # m=8: 16 + 28 = 44 (likely enough)
    
    # Let's try m from 6 to 12
    for m in range(6, 13):
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid set of size m
        found = False
        rng = random.Random(42 + m)
        
        # Greedy initialization
        S_greedy = set()
        covered = {N}
        for _ in range(m):
            best_k = -1
            best_gain = -1
            for k in candidates:
                if k in S_greedy: continue
                new_covered = covered | {k+1, N-k+1}
                # Add pair gains
                for other in S_greedy:
                    new_covered.add(abs(k-other)+2)
                gain = len(new_covered - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            if best_k == -1: break
            S_greedy.add(best_k)
            covered |= {best_k+1, N-best_k+1}
            for other in S_greedy - {best_k}:
                covered.add(abs(best_k-other)+2)
                
        if is_pancyclic(S_greedy):
            best_S = S_greedy
            found = True
            break
            
        # Local Search / Simulated Annealing
        # We want to minimize missing lengths.
        # Start with random sets or the greedy one.
        
        for restart in range(200): # 200 restarts per m
            if time.time() > deadline - 5.0:
                break
                
            if restart == 0:
                S = set(S_greedy)
            else:
                S = set(rng.sample(candidates, m))
                
            current_covered = get_covered_lengths(S)
            missing = [L for L in range(3, N) if L not in current_covered]
            current_cost = len(missing)
            
            if current_cost == 0:
                best_S = S
                found = True
                break
                
            T = 1.0
            for step in range(5000):
                if time.time() > deadline - 5.0:
                    break
                    
                # Pick a chord to remove and one to add
                k_out = rng.choice(list(S))
                k_in = rng.choice([k for k in candidates if k not in S])
                
                # Calculate new cost
                # Remove k_out contributions
                # Add k_in contributions
                
                # Simple recalculation is O(m^2) which is small.
                new_S = S.copy()
                new_S.remove(k_out)
                new_S.add(k_in)
                
                new_covered = get_covered_lengths(new_S)
                new_missing = [L for L in range(3, N) if L not in new_covered]
                new_cost = len(new_missing)
                
                delta = new_cost - current_cost
                
                if delta <= 0 or rng.random() < pow(2.718, -delta / max(T, 0.01)):
                    S = new_S
                    current_cost = new_cost
                    current_covered = new_covered
                    if current_cost == 0:
                        best_S = S
                        found = True
                        break
                T *= 0.99 # Cooling
                
            if found:
                break
                
        if found:
            break

    if best_S is None:
        # Fallback: Use a larger m, e.g., 12, with a simple construction
        # Or just use the greedy result from the last m tried
        # Let's construct a valid one with m=12 if we haven't found one
        # The greedy for m=12 should definitely work.
        S_greedy = set()
        covered = {N}
        candidates_list = list(range(2, N-1))
        for _ in range(12):
            best_k = -1
            best_gain = -1
            for k in candidates_list:
                if k in S_greedy: continue
                new_covered = covered | {k+1, N-k+1}
                for other in S_greedy:
                    new_covered.add(abs(k-other)+2)
                gain = len(new_covered - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            if best_k == -1: break
            S_greedy.add(best_k)
            covered |= {best_k+1, N-best_k+1}
            for other in S_greedy - {best_k}:
                covered.add(abs(best_k-other)+2)
        best_S = S_greedy
        
    result = build_result(best_S)
    if result is None:
        # Emergency fallback: Complete graph? No, too many edges.
        # Just return a valid sparse one, even if not minimal.
        # The greedy above should be valid.
        pass
        
    return result
