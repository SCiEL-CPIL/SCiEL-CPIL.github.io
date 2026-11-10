import time
import random
import math

N = 256
DEADLINE = 50.0

def _precompute():
    """
    Precompute data structures for 3-AP and 4-AP relationships.
    """
    ap3_pairs = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_pairs[a].append((b, c))
            ap3_pairs[b].append((a, c))
            ap3_pairs[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_pairs, ap4_partners

def _count_3aps(M, ap3_pairs):
    """
    Count 3-APs in the set represented by bitmask M.
    """
    count = 0
    for i in range(1, N + 1):
        if M & (1 << i):
            for j, k in ap3_pairs[i]:
                if (M & (1 << j)) and (M & (1 << k)):
                    count += 1
    return count // 3

def _is_valid_add(y, M, ap4_partners):
    """
    Check if adding y to set M creates a 4-AP.
    """
    for j, k, l in ap4_partners[y]:
        if (M & (1 << j)) and (M & (1 << k)) and (M & (1 << l)):
            return False
    return True

def _gain_add(y, M, ap3_pairs):
    """
    Calculate the number of 3-APs formed by adding y to set M.
    """
    gain = 0
    for j, k in ap3_pairs[y]:
        if (M & (1 << j)) and (M & (1 << k)):
            gain += 1
    return gain

def _loss_remove(x, M, ap3_pairs):
    """
    Calculate the number of 3-APs destroyed by removing x from set M.
    """
    loss = 0
    for j, k in ap3_pairs[x]:
        if (M & (1 << j)) and (M & (1 << k)):
            loss += 1
    return loss

def _greedy_init(ap3_pairs, ap4_partners, seed=None):
    """
    Generate an initial set using a greedy randomized approach.
    """
    if seed is not None:
        random.seed(seed)
        
    M = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    # First pass: Add elements that don't create 4-APs
    # We might want to prioritize elements with high potential 3-AP count
    # But for initialization, just valid adds are fine.
    for y in candidates:
        if _is_valid_add(y, M, ap4_partners):
            M |= (1 << y)
            
    return M

def _tabu_search(M_init, ap3_pairs, ap4_partners, time_limit, tabu_tenure=20):
    """
    Perform Tabu Search to maximize 3-APs.
    """
    start_time = time.time()
    M = M_init
    current_score = _count_3aps(M, ap3_pairs)
    best_score = current_score
    best_M = M
    
    # Tabu list stores (type, arg1, arg2)
    # type 0: add y
    # type 1: remove x
    # type 2: swap x, y
    tabu_list = []
    
    # Precompute list of present and absent elements for faster iteration
    # But since N=256, iterating 1..N is fine.
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        best_delta = -10**9
        best_move = None
        
        # 1. Try Swaps
        # Sample a subset of present and absent to keep it fast?
        # With N=256, full scan is 256*256 = 65536 checks.
        # Each check is O(degree_4) ~ 10-20. So ~1M ops.
        # This might be slow in Python if done naively in a tight loop.
        # Let's optimize by pre-filtering or sampling.
        
        present = [i for i in range(1, N + 1) if M & (1 << i)]
        absent = [i for i in range(1, N + 1) if not (M & (1 << i))]
        
        if not present or not absent:
            break
            
        # To speed up, we can sample if the set is large, but N is small.
        # Let's try all swaps but break early if time is tight.
        
        # Optimization: Only consider swaps where y has some gain potential
        # and x has some loss.
        
        for x in present:
            loss_x = _loss_remove(x, M, ap3_pairs)
            if loss_x == 0:
                # Removing x doesn't hurt 3-APs. This is a good candidate to remove.
                pass
            
            for y in absent:
                # Check tabu
                is_tabu = False
                for t_type, t_a1, t_a2 in tabu_list:
                    if t_type == 2 and ((t_a1 == x and t_a2 == y) or (t_a1 == y and t_a2 == x)):
                        is_tabu = True
                        break
                
                # Check validity: y added to M \ {x}
                M_temp = M & ~(1 << x)
                if not _is_valid_add(y, M_temp, ap4_partners):
                    continue
                
                gain_y = _gain_add(y, M_temp, ap3_pairs)
                delta = gain_y - loss_x
                
                # Aspiration: Accept if it improves global best
                if not is_tabu or (current_score + delta > best_score):
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (2, x, y)
                        
        # 2. Try Adds
        for y in absent:
            is_tabu = False
            for t_type, t_a1, t_a2 in tabu_list:
                if t_type == 0 and t_a1 == y:
                    is_tabu = True
                    break
                    
            if not _is_valid_add(y, M, ap4_partners):
                continue
                
            gain_y = _gain_add(y, M, ap3_pairs)
            delta = gain_y
            
            if not is_tabu or (current_score + delta > best_score):
                if delta > best_delta:
                    best_delta = delta
                    best_move = (0, y, 0)
                    
        # 3. Try Removes
        for x in present:
            is_tabu = False
            for t_type, t_a1, t_a2 in tabu_list:
                if t_type == 1 and t_a1 == x:
                    is_tabu = True
                    break
                    
            loss_x = _loss_remove(x, M, ap3_pairs)
            delta = -loss_x
            
            if not is_tabu or (current_score + delta > best_score):
                if delta > best_delta:
                    best_delta = delta
                    best_move = (1, x, 0)
                    
        if best_move is None:
            # No valid moves found
            # Try to relax tabu if stuck
            if no_improve_count > 20:
                tabu_list = []
                continue
            else:
                break
                
        # Apply best move
        if best_move[0] == 2: # Swap
            _, x, y = best_move
            M = (M & ~(1 << x)) | (1 << y)
            tabu_list.append((2, x, y))
        elif best_move[0] == 0: # Add
            _, y, _ = best_move
            M |= (1 << y)
            tabu_list.append((0, y, 0))
        elif best_move[0] == 1: # Remove
            _, x, _ = best_move
            M &= ~(1 << x)
            tabu_list.append((1, x, 0))
            
        current_score += best_delta
        
        if current_score > best_score:
            best_score = current_score
            best_M = M
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu list
        if len(tabu_list) >= tabu_tenure:
            tabu_list.pop(0)
            
    return best_score, best_M

def construct():
    start_time = time.time()
    ap3_pairs, ap4_partners = _precompute()
    
    best_score = -1
    best_M = 0
    
    # Run multiple restarts
    while time.time() - start_time < DEADLINE:
        # Generate a few diverse initial solutions
        for i in range(5):
            M_init = _greedy_init(ap3_pairs, ap4_partners, seed=random.randint(0, 1000000))
            
            # Refine with Tabu Search
            # Allocate time for this refinement
            time_left = DEADLINE - (time.time() - start_time)
            if time_left < 2.0:
                break
                
            # Use a fraction of remaining time for each refinement
            # Or just a fixed small amount to allow many restarts
            refine_time = min(5.0, time_left / 2.0)
            
            score, M = _tabu_search(M_init, ap3_pairs, ap4_partners, refine_time)
            
            if score > best_score:
                best_score = score
                best_M = M
                
    # Convert bitmask to list
    result = [i for i in range(1, N + 1) if best_M & (1 << i)]
    
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
