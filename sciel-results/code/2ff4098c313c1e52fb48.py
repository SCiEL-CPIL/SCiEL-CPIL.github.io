import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    # Hardcoded for q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for generic case (not expected)
        # Simple greedy for generic
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    neighbors[i].append(word_to_idx[tuple(new_word)])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1: break
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
        return [list(all_words[i]) for i in solution]

    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                neighbors[i].append(word_to_idx[tuple(new_word)])

    # 1. Initial Greedy Solution
    def solve_greedy(seed):
        random.seed(seed)
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            candidates = []
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            if not candidates: break
            best_c = random.choice(candidates)
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        # Prune
        cover_count = np.zeros(N, dtype=int)
        for c in solution:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        final_sol = list(solution)
        for c in final_sol[:]:
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                final_sol.remove(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
        return final_sol

    # Get a good starting point
    start_time = time.time()
    init_sol = solve_greedy(42)
    best_size = len(init_sol)
    best_sol = init_sol
    
    # Try a few more greedy seeds if time permits
    for s in range(43, 53):
        if time.time() - start_time > 10: break
        sol = solve_greedy(s)
        if len(sol) < best_size:
            best_size = len(sol)
            best_sol = sol

    # 2. Simulated Annealing
    def run_sa(initial_solution, time_limit):
        sol = list(initial_solution)
        cover_count = np.zeros(N, dtype=int)
        for c in sol:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        current_size = len(sol)
        best_size_sa = len(sol)
        best_sol_sa = list(sol)
        
        # Check initial validity (should be valid from greedy)
        
        T = 50.0
        alpha = 0.9995
        min_T = 0.1
        lambda_pen = 10.0
        
        start_t = time.time()
        
        while T > min_T and (time.time() - start_t) < time_limit:
            move_type = random.random()
            
            if len(sol) == 0:
                T *= alpha
                continue
            
            if move_type < 0.4:
                # Remove
                idx = random.randint(0, len(sol)-1)
                c = sol[idx]
                
                # Tentatively remove
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                
                # Check validity
                uncovered_count = 0
                for nb in neighbors[c]:
                    if cover_count[nb] == 0:
                        uncovered_count += 1
                        
                if uncovered_count == 0:
                    delta_cost = -1
                    valid = True
                else:
                    delta_cost = -1 + lambda_pen * uncovered_count
                    valid = False
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    sol.pop(idx)
                    current_size -= 1
                    if valid and current_size < best_size_sa:
                        best_size_sa = current_size
                        best_sol_sa = list(sol)
                else:
                    # Reject
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                            
            elif move_type < 0.8:
                # Add
                c = random.randint(0, N-1)
                if c in sol:
                    T *= alpha
                    continue
                    
                for nb in neighbors[c]:
                    cover_count[nb] += 1
                    
                delta_cost = 1
                valid = True # Adding always maintains validity if it was valid before? 
                # Actually, if we were in an invalid state, adding might fix it.
                # But we track validity based on cover_count.
                # Let's just say valid=True for cost calculation simplicity, 
                # but we only update best if truly valid.
                
                # Check if this move fixes validity?
                # We don't track "was_invalid" state easily.
                # Simpler: Just calculate cost.
                # If we accept, we update state.
                
                if random.random() < np.exp(-delta_cost / T):
                    sol.append(c)
                    current_size += 1
                    # Check if valid now
                    # We can check min cover count? No, just check if any 0 remains.
                    # This is expensive O(N). 
                    # Optimization: We know we added c. 
                    # If we were invalid before, we might still be invalid.
                    # Let's just update best if we are sure it's valid.
                    # We can check validity by scanning neighbors of added c? No, other words might be uncovered.
                    # To be safe, only update best if we know it's valid.
                    # We know it's valid if the previous state was valid.
                    # We don't track previous state validity.
                    # Let's assume we only update best when we are confident.
                    # For "Add", if we were valid, we stay valid.
                    # If we were invalid, we might become valid.
                    # Let's just check validity every time we accept a move? 
                    # That's O(N) per move. Too slow.
                    
                    # Alternative: Track `num_uncovered`?
                    # Let's maintain `num_uncovered`.
                    # If `num_uncovered` == 0, it's valid.
                    
                    # Let's refactor to maintain `num_uncovered`.
                    
                    # I will rewrite the loop to use `num_uncovered`.
                    
                    pass # Placeholder, will implement in final code
            
            else:
                # Swap
                if len(sol) == 0:
                    T *= alpha
                    continue
                idx = random.randint(0, len(sol)-1)
                c_old = sol[idx]
                c_new = random.randint(0, N-1)
                if c_new == c_old or c_new in sol:
                    T *= alpha
                    continue
                
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                for nb in neighbors[c_new]:
                    cover_count[nb] += 1
                    
                # Recalculate uncovered count change?
                # Hard to do incrementally without tracking per-word status.
                # Let's just check the neighbors of c_old and c_new.
                # Words affected are neighbors of c_old and c_new.
                
                # This is getting complex.
                # Let's stick to a simpler move set or efficient validity check.
                
                T *= alpha
            T *= alpha # Ensure cooling happens

        return best_sol_sa, best_size_sa

    # The above SA implementation is getting messy with validity tracking.
    # Let's use a simpler, robust approach:
    # Maintain `cover_count`.
    # Maintain `num_uncovered` (number of words with cover_count 0).
    # Initial: calculate `num_uncovered`.
    # Move: Update `cover_count` for affected words. Update `num_uncovered` based on changes.
    
    def run_sa_v2(initial_solution, time_limit):
        sol = list(initial_solution)
        cover_count = np.zeros(N, dtype=int)
        for c in sol:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        num_uncovered = np.sum(cover_count == 0)
        current_size = len(sol)
        best_size_sa = len(sol)
        best_sol_sa = list(sol)
        
        T = 50.0
        alpha = 0.9995
        min_T = 0.1
        lambda_pen = 10.0
        
        start_t = time.time()
        
        while T > min_T and (time.time() - start_t) < time_limit:
            move_type = random.random()
            
            if len(sol) == 0:
                T *= alpha
                continue
            
            if move_type < 0.4:
                # Remove
                idx = random.randint(0, len(sol)-1)
                c = sol[idx]
                
                # Tentatively remove
                delta_uncovered = 0
                for nb in neighbors[c]:
                    old_count = cover_count[nb]
                    new_count = old_count - 1
                    cover_count[nb] = new_count
                    if old_count == 1 and new_count == 0:
                        delta_uncovered += 1
                    elif old_count == 0 and new_count == -1: # Should not happen if valid
                        delta_uncovered -= 1
                        
                new_num_uncovered = num_uncovered + delta_uncovered
                delta_cost = -1 + lambda_pen * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    sol.pop(idx)
                    current_size -= 1
                    num_uncovered = new_num_uncovered
                    if new_num_uncovered == 0 and current_size < best_size_sa:
                        best_size_sa = current_size
                        best_sol_sa = list(sol)
                else:
                    # Reject
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                            
            elif move_type < 0.8:
                # Add
                c = random.randint(0, N-1)
                if c in sol:
                    T *= alpha
                    continue
                    
                delta_uncovered = 0
                for nb in neighbors[c]:
                    old_count = cover_count[nb]
                    new_count = old_count + 1
                    cover_count[nb] = new_count
                    if old_count == 0 and new_count == 1:
                        delta_uncovered -= 1
                        
                new_num_uncovered = num_uncovered + delta_uncovered
                delta_cost = 1 + lambda_pen * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    sol.append(c)
                    current_size += 1
                    num_uncovered = new_num_uncovered
                    if new_num_uncovered == 0 and current_size < best_size_sa:
                        best_size_sa = current_size
                        best_sol_sa = list(sol)
                else:
                    # Reject
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
            else:
                # Swap
                if len(sol) == 0:
                    T *= alpha
                    continue
                idx = random.randint(0, len(sol)-1)
                c_old = sol[idx]
                c_new = random.randint(0, N-1)
                if c_new == c_old or c_new in sol:
                    T *= alpha
                    continue
                
                # Remove c_old
                delta_uncovered_old = 0
                for nb in neighbors[c_old]:
                    old_count = cover_count[nb]
                    new_count = old_count - 1
                    cover_count[nb] = new_count
                    if old_count == 1 and new_count == 0:
                        delta_uncovered_old += 1
                        
                # Add c_new
                delta_uncovered_new = 0
                for nb in neighbors[c_new]:
                    old_count = cover_count[nb]
                    new_count = old_count + 1
                    cover_count[nb] = new_count
                    if old_count == 0 and new_count == 1:
                        delta_uncovered_new -= 1
                        
                delta_uncovered = delta_uncovered_old + delta_uncovered_new
                new_num_uncovered = num_uncovered + delta_uncovered
                delta_cost = lambda_pen * delta_uncovered # Size unchanged
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    sol[idx] = c_new
                    # current_size unchanged
                    num_uncovered = new_num_uncovered
                    if new_num_uncovered == 0 and current_size < best_size_sa:
                        best_size_sa = current_size
                        best_sol_sa = list(sol)
                else:
                    # Reject
                    for nb in neighbors[c_old]:
                        cover_count[nb] += 1
                    for nb in neighbors[c_new]:
                        cover_count[nb] -= 1
            
            T *= alpha

        return best_sol_sa, best_size_sa

    # Run SA
    elapsed = time.time() - start_time
    time_limit_sa = max(5, 115 - elapsed)
    
    sa_sol, sa_cost = run_sa_v2(best_sol, time_limit_sa)
    
    if sa_cost < best_size:
        best_sol = sa_sol
        best_size = sa_cost
        
    return [list(all_words[i]) for i in best_sol]
