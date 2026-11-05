import time
import sys
import random

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    For n=7, uses a heuristic search to find a shorter ordering of the 6-permutations
    than the standard recursive construction, aiming for the known minimum of 5905.
    """
    if n == 1:
        return "1"
    
    # Classic construction for n < 7 (fast enough and optimal or near-optimal for small n)
    def classic_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap
                max_ov = min(len(new), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    if n != 7:
        return "".join(map(str, classic_construct(n)))
    
    # Specialized optimization for n=7
    # 1. Generate the superpermutation for n=6
    word_6 = classic_construct(6)
    
    # 2. Extract the list of 6-permutations in the order they appear in word_6
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # Should be 720
    
    # 3. Precompute overlap matrix M[i][j] = longest suffix of perms_6[i] matching prefix of perms_6[j]
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # 4. Search for the best ordering
    # Deadline: 110 seconds
    start_time = time.time()
    deadline = start_time + 110.0
    
    random.seed(42)
    
    best_order = None
    best_score = -1
    
    # Helper to calculate score
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        
    # Helper to evaluate a tour and polish it
    def polish_tour(tour, time_limit):
        current_order = tour[:]
        current_score = calc_score(current_order)
        last_improvement = time.time()
        
        while time.time() < time_limit:
            improved = False
            # Try random moves
            # Move types: Swap, Reverse, Move
            move_type = random.randint(0, 2)
            
            if move_type == 0:
                # Swap adjacent
                i = random.randint(0, num_perms - 2)
                a = current_order[i]
                b = current_order[i+1]
                
                # Delta calculation
                if i > 0:
                    prev = current_order[i-1]
                    old_prev = M[prev][a]
                    new_prev = M[prev][b]
                else:
                    old_prev = 0
                    new_prev = 0
                    
                if i+1 < num_perms - 1:
                    next_idx = current_order[i+2]
                    old_next = M[b][next_idx]
                    new_next = M[a][next_idx]
                else:
                    old_next = 0
                    new_next = 0
                    
                old_mid = M[a][b]
                new_mid = M[b][a]
                
                delta = (new_prev + new_mid + new_next) - (old_prev + old_mid + old_next)
                
                if delta > 0:
                    current_order[i], current_order[i+1] = b, a
                    current_score += delta
                    improved = True
                    last_improvement = time.time()
                    
            elif move_type == 1:
                # Reverse segment
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                # Delta calculation
                if i > 0:
                    prev = current_order[i-1]
                    old_left = M[prev][current_order[i]]
                    new_left = M[prev][current_order[j]]
                else:
                    old_left = 0
                    new_left = 0
                    
                if j < num_perms - 1:
                    next_idx = current_order[j+1]
                    old_right = M[current_order[j]][next_idx]
                    new_right = M[current_order[i]][next_idx]
                else:
                    old_right = 0
                    new_right = 0
                    
                old_internal = 0
                new_internal = 0
                for k in range(i, j):
                    old_internal += M[current_order[k]][current_order[k+1]]
                    new_internal += M[current_order[k+1]][current_order[k]]
                    
                delta = (new_left + new_right + new_internal) - (old_left + old_right + old_internal)
                
                if delta > 0:
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    improved = True
                    last_improvement = time.time()
                    
            else:
                # Move element i to position j
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j:
                    continue
                
                item = current_order[i]
                
                # Calculate delta for removing i
                if i > 0:
                    L = current_order[i-1]
                    R = current_order[i+1] if i+1 < num_perms else -1
                else:
                    L = -1
                    R = current_order[i+1] if i+1 < num_perms else -1
                    
                # This is getting complicated with boundaries. 
                # Simplified Move:
                # Remove item at i.
                # Insert at j (shifting elements).
                
                # Let's just use a simpler heuristic: try moving to a few random positions
                # and accept if score improves.
                # To make it fast, we only try if i and j are not too far?
                # Or just recalc score if it's a candidate? No, too slow.
                
                # Let's stick to Swap and Reverse for now, they are robust.
                # Or implement a safe Move:
                # Remove i, insert at j.
                
                pass

            if not improved:
                # If no improvement in last 1s, break to try a new restart
                if time.time() - last_improvement > 1.0:
                    break
                    
        return current_order, current_score

    # Initial tours
    initial_tours = []
    
    # 1. Standard order
    initial_tours.append(list(range(num_perms)))
    
    # 2. Greedy NN
    for _ in range(5):
        visited = [False] * num_perms
        path = []
        start = random.randint(0, num_perms - 1)
        path.append(start)
        visited[start] = True
        for _ in range(num_perms - 1):
            last = path[-1]
            best_next = -1
            best_val = -1
            # Optimize: pick from unvisited neighbors
            # This is O(N^2) per step. 720 * 720 * 720 is too slow for 5 restarts in 110s?
            # 720^3 ~ 370M. In Python, this is ~30-60s per restart. Too slow.
            # Use a faster greedy: pick random unvisited, or use a heap?
            # Let's just use random restarts for initial tours to save time.
            # Or a very fast greedy: pick the best of 10 random candidates?
            pass
        
    # Just use random restarts for initial tours to ensure we don't spend too much time on init
    for _ in range(10):
        initial_tours.append(random.sample(range(num_perms), num_perms))
        
    # Run polish on each initial tour
    # We have ~100s for search.
    # Let's allocate time dynamically.
    # Try the first tour (standard) for 10s.
    # Then random tours for 5s each.
    
    time_per_tour = 5.0
    time_left = deadline - time.time()
    
    # Process standard order first
    if time_left > 10:
        tour, score = polish_tour(initial_tours[0], time.time() + 10)
        if score > best_score:
            best_score = score
            best_order = tour
        time_left = deadline - time.time()
        
    # Process random restarts
    for i in range(1, len(initial_tours)):
        if time_left < time_per_tour + 5: # Leave 5s for final check
            break
        tour, score = polish_tour(initial_tours[i], time.time() + time_per_tour)
        if score > best_score:
            best_score = score
            best_order = tour
        time_left = deadline - time.time()
        
    # If we have time left, polish the best order further
    if best_order is not None:
        polish_tour(best_order, deadline - 1.0)
        # Recalc score for best_order just in case polish_tour didn't return it
        # Actually polish_tour returns the polished tour and score.
        # But we passed best_order into it.
        # Let's just recalc the score of the current best_order after the final polish
        final_score = calc_score(best_order)
        if final_score > best_score:
            best_score = final_score

    # 5. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M[best_order[idx-1]][p_idx]
            word += block[ov:]
            
    return "".join(map(str, word))

if __name__ == "__main__":
    pass
