"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i

    def get_score(cycle):
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos + 1

    def two_opt_refine(cycle, time_limit=85):
        """
        Refine a cycle using 2-opt moves to minimize the maximum edge index.
        """
        best_cycle = cycle[:]
        best_score = get_score(best_cycle)
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 2, n):
                    if time.time() - start_time > time_limit:
                        break
                    u1 = best_cycle[i]
                    v1 = best_cycle[(i + 1) % n]
                    u2 = best_cycle[j]
                    v2 = best_cycle[(j + 1) % n]
                    
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue

                    pos_old1 = edge_pos[(u1, v1)]
                    pos_old2 = edge_pos[(u2, v2)]
                    pos_new1 = edge_pos[(u1, u2)]
                    pos_new2 = edge_pos[(v1, v2)]
                    
                    old_max = max(pos_old1, pos_old2)
                    new_max = max(pos_new1, pos_new2)
                    
                    if new_max < old_max:
                        segment = best_cycle[i+1:j+1]
                        best_cycle[i+1:j+1] = reversed(segment)
                        new_score = get_score(best_cycle)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_cycle[i+1:j+1] = reversed(best_cycle[i+1:j+1])
        return best_cycle

    def simulated_annealing(cycle, time_limit=85, initial_temp=10.0, cooling=0.995):
        """
        Simulated Annealing with 2-opt moves to escape local minima.
        """
        current_cycle = cycle[:]
        current_score = get_score(current_cycle)
        best_cycle = current_cycle[:]
        best_score = current_score
        temp = initial_temp
        
        while temp > 0.01 and time.time() - start_time < time_limit:
            # Generate a random 2-opt move
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or abs(i - j) <= 1:
                continue
            
            # Ensure i < j for consistent reversal
            if i > j:
                i, j = j, i
            
            u1 = current_cycle[i]
            v1 = current_cycle[(i + 1) % n]
            u2 = current_cycle[j]
            v2 = current_cycle[(j + 1) % n]
            
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue

            # Perform the move
            segment = current_cycle[i+1:j+1]
            current_cycle[i+1:j+1] = reversed(segment)
            
            new_score = get_score(current_cycle)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = current_cycle[:]
            else:
                # Revert
                current_cycle[i+1:j+1] = reversed(current_cycle[i+1:j+1])
            
            temp *= cooling
        
        return best_cycle, best_score

    def greedy_build():
        """
        Greedy construction: build cycle by always picking the next vertex
        that minimizes the maximum edge position so far.
        """
        visited = [False] * n
        cycle = [0]
        visited[0] = True
        current_max = 0
        
        for _ in range(n - 1):
            last = cycle[-1]
            best_next = -1
            best_max = float('inf')
            
            for v in range(n):
                if not visited[v]:
                    pos = edge_pos[(last, v)]
                    new_max = max(current_max, pos)
                    if new_max < best_max:
                        best_max = new_max
                        best_next = v
            
            if best_next == -1:
                break
            
            cycle.append(best_next)
            visited[best_next] = True
            current_max = best_max
        
        # Close the cycle
        if len(cycle) == n:
            pos = edge_pos[(cycle[-1], cycle[0])]
            current_max = max(current_max, pos)
        
        return cycle, current_max + 1

    best_cycle = None
    best_score = float('inf')
    
    # Step 1: Greedy construction as initial solution
    if time.time() - start_time < 5:
        greedy_cycle, greedy_score = greedy_build()
        if greedy_score < best_score:
            best_score = greedy_score
            best_cycle = greedy_cycle
    
    # Step 2: 2-opt refinement
    if best_cycle and time.time() - start_time < 20:
        refined_cycle = two_opt_refine(best_cycle, time_limit=20)
        refined_score = get_score(refined_cycle)
        if refined_score < best_score:
            best_score = refined_score
            best_cycle = refined_cycle
    
    # Step 3: Simulated Annealing for more exploration
    if best_cycle and time.time() - start_time < 60:
        sa_cycle, sa_score = simulated_annealing(best_cycle, time_limit=60)
        if sa_score < best_score:
            best_score = sa_score
            best_cycle = sa_cycle
    
    # Step 4: Final 2-opt polish
    if best_cycle and time.time() - start_time < 85:
        final_cycle = two_opt_refine(best_cycle, time_limit=85)
        final_score = get_score(final_cycle)
        if final_score < best_score:
            best_score = final_score
            best_cycle = final_cycle
    
    # Fallback: return identity if something went wrong
    if best_cycle is None:
        best_cycle = list(range(n))
        best_score = get_score(best_cycle)
    
    return best_cycle

# EVOLVE-BLOCK-END
