import random
import math

def construct():
    n = 128
    target_min = 128
    target_max = 256
    
    # Precompute target range indices
    # counts array size 257 (0 to 256)
    
    def get_initial_counts(A):
        counts = [0] * (target_max + 1)
        k = len(A)
        for i in range(k):
            for j in range(i, k):
                s = A[i] + A[j]
                if target_min <= s <= target_max:
                    counts[s] += 1
        return counts

    def calculate_cost(counts):
        max_c = 0
        missing = 0
        for s in range(target_min, target_max + 1):
            c = counts[s]
            if c == 0:
                missing += 1
            elif c > max_c:
                max_c = c
        if missing > 0:
            return 1000 * missing + max_c
        return max_c

    def anneal(initial_A, steps=50000, T_start=1.0, T_end=0.01):
        current_A = list(initial_A)
        current_counts = get_initial_counts(current_A)
        current_cost = calculate_cost(current_counts)
        
        best_A = current_A.copy()
        best_cost = current_cost
        best_counts = current_counts.copy()
        
        k = len(current_A)
        
        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / steps)
            
            # Pick random index in A
            idx = random.randint(0, k - 1)
            old_val = current_A[idx]
            
            # Pick random value not in A
            # Retry loop to ensure new_val not in A
            new_val = random.randint(0, n)
            if new_val in current_A:
                # Simple retry, max 5 times
                for _ in range(5):
                    new_val = random.randint(0, n)
                    if new_val not in current_A:
                        break
                else:
                    continue # Still in A, skip
            
            if new_val == old_val:
                continue

            # Apply swap to counts
            # 1. Remove old_val
            # Pair (old_val, old_val)
            s_oo = 2 * old_val
            if target_min <= s_oo <= target_max:
                current_counts[s_oo] -= 1
            
            # Pairs (old_val, x) for x in A, x != old_val
            for x in current_A:
                if x == old_val:
                    continue
                s_ox = old_val + x
                if target_min <= s_ox <= target_max:
                    current_counts[s_ox] -= 1
                    
            # 2. Add new_val
            # Pair (new_val, new_val)
            s_nn = 2 * new_val
            if target_min <= s_nn <= target_max:
                current_counts[s_nn] += 1
                
            # Pairs (new_val, x) for x in A, x != old_val (since old_val is removed, new_val is added)
            # Note: current_A still contains old_val. We are effectively swapping.
            # The set becomes (A \ {old_val}) U {new_val}.
            # So x should range over A \ {old_val}.
            for x in current_A:
                if x == old_val:
                    continue
                s_nx = new_val + x
                if target_min <= s_nx <= target_max:
                    current_counts[s_nx] += 1
                    
            # Update A
            current_A[idx] = new_val
            
            # Calculate new cost
            new_cost = calculate_cost(current_counts)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A.copy()
                    best_counts = current_counts.copy()
            else:
                # Revert
                current_A[idx] = old_val
                
                # Revert counts
                # 1. Remove new_val effects
                s_nn = 2 * new_val
                if target_min <= s_nn <= target_max:
                    current_counts[s_nn] -= 1
                for x in current_A:
                    if x == old_val:
                        continue
                    s_nx = new_val + x
                    if target_min <= s_nx <= target_max:
                        current_counts[s_nx] -= 1
                        
                # 2. Add old_val effects back
                s_oo = 2 * old_val
                if target_min <= s_oo <= target_max:
                    current_counts[s_oo] += 1
                for x in current_A:
                    if x == old_val:
                        continue
                    s_ox = old_val + x
                    if target_min <= s_ox <= target_max:
                        current_counts[s_ox] += 1

        return best_A, best_cost

    best_overall_A = None
    best_overall_cost = float('inf')
    
    random.seed(42)
    
    # Try different sizes k
    for k in range(16, 20):
        for restart in range(10):
            A_init = random.sample(range(n + 1), k)
            A_found, cost = anneal(A_init, steps=10000, T_start=1.0, T_end=0.01)
            
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = A_found
                
            if best_overall_cost == 1:
                break
        if best_overall_cost == 1:
            break
            
    if best_overall_A is None:
        return list(range(64, 129))
        
    return best_overall_A
