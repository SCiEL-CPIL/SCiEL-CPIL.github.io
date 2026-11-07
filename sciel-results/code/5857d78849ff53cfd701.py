"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    best_set = []
    best_size = 0
    
    # Iterate possible starting pairs
    # x1 and x2 must be small enough to allow room for more elements
    # but large enough to satisfy constraints?
    # Actually, smaller x1, x2 allows smaller x3, etc.
    # But x1+x2 bounds x_M.
    # If x1+x2 is small, max size is small.
    # If x1+x2 is large, we have more room, but constraints might be tighter?
    
    # Let's try all x1, x2
    for x1 in range(1, 129):
        for x2 in range(x1 + 1, 257):
            if time.time() > deadline:
                break
                
            current_set = [x1, x2]
            
            # Greedily add next element
            while True:
                next_x = current_set[-1] + 1
                valid = True
                
                # Check constraints for adding next_x
                # New size M = len(current_set) + 1
                M = len(current_set) + 1
                
                if next_x > 256:
                    break
                    
                # Check all k in [2, M-1]
                # Condition: sum_{j=1}^k x_j > sum_{j=M-k+2}^{M-1} x_j + next_x
                # Rearranged: next_x < sum_{j=1}^k x_j - sum_{j=M-k+2}^{M-1} x_j
                
                # Precompute prefix sums for current_set
                # current_set has size M-1
                # We need sums of first k elements
                # and sums of last (M-k-1) elements?
                # T_{M,k} = sum_{j=M-k+2}^{M-1} x_j
                # Note indices in current_set are 0 to M-2.
                # j in 1-based for x_j.
                # j ranges from M-k+2 to M-1.
                # In 0-based: from M-k+1 to M-2.
                
                # Optimization:
                # The most restrictive k is usually small.
                # k=2: next_x < x1 + x2
                # k=3: next_x < x1+x2+x3 - x_{M-1} (if M-1 >= 3)
                
                for k in range(2, M):
                    # Sum of first k elements
                    sum_first_k = sum(current_set[:k])
                    
                    # Sum of elements from index M-k+1 to M-2 (0-based)
                    # Count of elements: (M-2) - (M-k+1) + 1 = k-2
                    if k == 2:
                        sum_tail = 0
                    else:
                        # Indices: M-k+1 to M-2
                        # Example M=4, k=3: Indices 4-3+1=2 to 2. x[2].
                        # current_set size 3. x[2] is last.
                        sum_tail = sum(current_set[M-k+1 : M-1])
                    
                    if next_x >= sum_first_k - sum_tail:
                        valid = False
                        break
                
                if valid:
                    current_set.append(next_x)
                else:
                    break
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
        if time.time() > deadline:
            break
            
    # Fallback if no set found (shouldn't happen)
    if not best_set:
        best_set = [1, 2]
        
    return best_set
# EVOLVE-BLOCK-END
