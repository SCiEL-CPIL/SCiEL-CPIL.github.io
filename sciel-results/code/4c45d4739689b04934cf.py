import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 4-APs in [1, N]
    all_4aps = []  # list of (a, d) for each 4-AP
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            all_4aps.append((a, d))
    
    # For each element x, precompute which 4-APs it participates in
    # and its position in each
    elem_4aps = {x: [] for x in range(1, N + 1)}
    for (a, d) in all_4aps:
        terms = [a, a + d, a + 2*d, a + 3*d]
        for pos, t in enumerate(terms):
            elem_4aps[t].append((a, d, pos))
    
    # Precompute all 3-APs
    all_3aps = []  # list of (a, d) for each 3-AP
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            all_3aps.append((a, d))
    
    # For each element x, precompute which 3-APs it participates in
    elem_3aps = {x: [] for x in range(1, N + 1)}
    for (a, d) in all_3aps:
        terms = [a, a + d, a + 2*d]
        for pos, t in enumerate(terms):
            elem_3aps[t].append((a, d, pos))
    
    def count_3aps(S_set):
        """Count distinct 3-APs in set S."""
        count = 0
        for (a, d) in all_3aps:
            if a in S_set and a + d in S_set and a + 2*d in S_set:
                count += 1
        return count
    
    def has_4ap(S_set):
        """Check if set S contains a nonconstant 4-term AP."""
        for (a, d) in all_4aps:
            if a in S_set and a + d in S_set and a + 2*d in S_set and a + 3*d in S_set:
                return True
        return False
    
    def get_4aps_created_by_add(x, S_set):
        """Get 4-APs that would be created by adding x to S_set."""
        created = []
        for (a, d, pos) in elem_4aps[x]:
            terms = [a, a + d, a + 2*d, a + 3*d]
            # Check if the other 3 terms are in S_set
            others = [terms[i] for i in range(4) if i != pos]
            if all(t in S_set for t in others):
                created.append((a, d))
        return created
    
    def get_3aps_gained_by_add(x, S_set):
        """Count 3-APs gained by adding x to S_set."""
        gained = 0
        for (a, d, pos) in elem_3aps[x]:
            terms = [a, a + d, a + 2*d]
            others = [terms[i] for i in range(3) if i != pos]
            if all(t in S_set for t in others):
                gained += 1
        return gained
    
    def get_3aps_lost_by_remove(x, S_set):
        """Count 3-APs lost by removing x from S_set."""
        lost = 0
        for (a, d, pos) in elem_3aps[x]:
            terms = [a, a + d, a + 2*d]
            others = [terms[i] for i in range(3) if i != pos]
            if all(t in S_set for t in others):
                lost += 1
        return lost
    
    def greedy_construction(randomize=True):
        """Build a 4-AP-free set greedily."""
        S = set()
        candidates = list(range(1, N + 1))
        if randomize:
            random.shuffle(candidates)
        
        for x in candidates:
            if not get_4aps_created_by_add(x, S):
                S.add(x)
        
        return S
    
    def local_search(S, max_iter=5000):
        """Local search with tabu memory."""
        S_set = set(S)
        current_score = count_3aps(S_set)
        best_S = set(S_set)
        best_score = current_score
        
        tabu_list = {}
        tabu_tenure = 15
        iteration = 0
        
        while iteration < max_iter:
            if time.time() > DEADLINE:
                break
            
            best_move = None
            best_move_score = current_score
            best_move_type = None
            best_move_elem = None
            
            # Try adding elements
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if tabu_list.get(x, 0) > iteration:
                    continue
                if get_4aps_created_by_add(x, S_set):
                    continue
                gain = get_3aps_gained_by_add(x, S_set)
                new_score = current_score + gain
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'add'
                    best_move_elem = x
                    best_move = (x, new_score)
            
            # Try removing elements
            for x in S_set:
                if tabu_list.get(x, 0) > iteration:
                    continue
                loss = get_3aps_lost_by_remove(x, S_set)
                new_score = current_score - loss
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'remove'
                    best_move_elem = x
                    best_move = (x, new_score)
            
            if best_move is None:
                break
            
            # Apply best move
            if best_move_type == 'add':
                S_set.add(best_move_elem)
                tabu_list[best_move_elem] = iteration + tabu_tenure
            elif best_move_type == 'remove':
                S_set.discard(best_move_elem)
                tabu_list[best_move_elem] = iteration + tabu_tenure
            
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S_set)
            
            iteration += 1
        
        return best_S, best_score
    
    best_overall = set()
    best_overall_score = 0
    
    # First, try a deterministic greedy
    S0 = greedy_construction(randomize=False)
    best_S, best_score = local_search(S0, max_iter=3000)
    best_overall = best_S
    best_overall_score = best_score
    
    # Try randomized greedy constructions
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        S_i = greedy_construction(randomize=True)
        best_S, best_score = local_search(S_i, max_iter=2000)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    # Try Cantor-like set
    S_cantor = set()
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 4 >= 2:
                valid = False
                break
            t //= 4
        if valid:
            S_cantor.add(x)
    
    if S_cantor and not has_4ap(S_cantor):
        best_S, best_score = local_search(S_cantor, max_iter=3000)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    # Final: try a few more random restarts if time permits
    while time.time() < DEADLINE - 10:
        S_rand = greedy_construction(randomize=True)
        best_S, best_score = local_search(S_rand, max_iter=1500)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S
    
    return sorted(best_overall)
