"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # 1. Initial Solution: Rank Matching
    # Assign facilities with high flow sums to locations with low distance sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    # Calculate initial cost
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    best_p = p.copy()
    best_cost = calculate_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # Precompute symmetric parts for speed if matrices are symmetric (QAPLIB standard)
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if a_sym and b_sym:
        # Use efficient delta calculation
        # Precompute diagonal terms
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        # Tabu Search Structures
        tabu_list = {}  # (i, j) -> time when tabu expires (iteration count)
        tabu_tenure = 100
        iteration = 0
        
        # Frequency memory for diversification
        freq_i = np.zeros(n, dtype=np.int64)
        freq_j = np.zeros(n, dtype=np.int64)
        
        # Helper to calculate delta for swapping i and j
        def calc_delta(i, j, perm):
            u = perm[i]
            v = perm[j]
            
            # Diagonal part
            delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
            
            # Off-diagonal part
            diff_A = a[i, :] - a[j, :]
            diff_B = b[v, perm] - b[u, perm]
            
            total_dot = np.dot(diff_A, diff_B)
            
            # Exclude i and j
            exclude_i = diff_A[i] * diff_B[i]
            expire_j = diff_A[j] * diff_B[j]
            
            off_diag_sum = total_dot - exclude_i - expire_j
            delta += 2 * off_diag_sum
            
            return int(delta)

        # Main Loop
        while time.time() - start_time < time_limit:
            improved = False
            best_delta_in_pass = 0
            best_move_i = -1
            best_move_j = -1
            
            # Generate a random order of pairs to avoid bias
            # For n=256, there are ~32k pairs. Shuffling all is expensive.
            # We can just iterate but maybe randomize start? Or just standard.
            # Let's stick to standard for speed, but rely on tabu to escape.
            
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                u = current_p[i]
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    v = current_p[j]
                    
                    # Check tabu
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)
                    
                    delta = calc_delta(i, j, current_p)
                    
                    # Aspiration: accept if it improves global best even if tabu
                    if delta < 0 or (is_tabu and current_cost + delta < best_cost):
                        # Perform swap
                        current_p[i], current_p[j] = v, u
                        current_cost += delta
                        
                        # Update tabu
                        tabu_list[(i, j)] = iteration + tabu_tenure
                        tabu_list[(j, i)] = iteration + tabu_tenure
                        
                        # Update frequency
                        freq_i[i] += 1
                        freq_i[j] += 1
                        freq_j[i] += 1
                        freq_j[j] += 1
                        
                        # Update best
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                        
                        improved = True
                        # Break out of inner loop to restart pass?
                        # Standard tabu often does one move per pass or continues.
                        # Let's continue scanning but mark as improved.
                        # Actually, for speed, if we found a good move, maybe break?
                        # No, let's just take the first improving move to keep it fast.
                        break 
                
                if improved:
                    break # Break outer loop too, restart pass
                    
            iteration += 1
            
            if not improved:
                # Local optimum reached (no improving non-tabu moves)
                # Diversification: Random kick
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                # Choose a random move, even if it increases cost
                # Weighted by frequency to escape high-frequency areas
                # Simple random kick for now
                i_kick = random.randint(0, n-1)
                j_kick = random.randint(0, n-1)
                if i_kick != j_kick:
                    delta = calc_delta(i_kick, j_kick, current_p)
                    current_p[i_kick], current_p[j_kick] = current_p[j_kick], current_p[i_kick]
                    current_cost += delta
                    
                    # Update tabu
                    tabu_list[(i_kick, j_kick)] = iteration + tabu_tenure
                    tabu_list[(j_kick, i_kick)] = iteration + tabu_tenure
                    
                    # Update best if improved (unlikely)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    
                    # Clear some tabu memory to allow backtracking
                    # Or just let it expire
                    pass
            else:
                # We found an improving move, so we continued.
                # We break out of the pass loop after the first improvement.
                pass

    else:
        # Fallback for non-symmetric: return initial solution
        pass

    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
