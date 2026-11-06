"""
Optimization program for maximizing 3-term APs in a subset of [1,256] with no 4-term AP.
Strategy: Iterated Local Search (ILS) with Simulated Annealing on the local optima.
1. Fast Local Search (Add/Swap moves) to convergence.
2. Perturbation to escape local optima.
3. Acceptance criterion based on improvement or simulated annealing probability.
"""
import time
import random
import math

N = 256
MAX_TIME = 85  # seconds, safety margin before 90s kill

# Precompute 3-APs and 4-APs
_three_aps = []
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        _three_aps.append((a, a + d, a + 2 * d))

_four_aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        _four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

# Map elements to APs containing them
_elem_to_3aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_three_aps):
    for x in ap:
        _elem_to_3aps[x].append(i)

_elem_to_4aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_four_aps):
    for x in ap:
        _elem_to_4aps[x].append(i)


def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in set S."""
    S = set(S)
    count = 0
    for ap in _three_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S:
            count += 1
    return count


def has_4ap(S):
    """Check if set S contains any nonconstant 4-term arithmetic progression."""
    S = set(S)
    for ap in _four_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S and ap[3] in S:
            return True
    return False


def is_valid_add(S, x):
    """Check if adding x to S creates a 4-AP."""
    for ap_idx in _elem_to_4aps[x]:
        a, b, c, d = _four_aps[ap_idx]
        if x == a:
            if b in S and c in S and d in S:
                return False
        elif x == b:
            if a in S and c in S and d in S:
                return False
        elif x == c:
            if a in S and b in S and d in S:
                return False
        else:
            if a in S and b in S and c in S:
                return False
    return True


def delta_add(S, x):
    """Change in 3-AP count if we add x to S."""
    delta = 0
    for ap_idx in _elem_to_3aps[x]:
        a, b, c = _three_aps[ap_idx]
        if x == a:
            if b in S and c in S:
                delta += 1
        elif x == b:
            if a in S and c in S:
                delta += 1
        else:
            if a in S and b in S:
                delta += 1
    return delta


def is_valid_swap(S, x_out, x_in):
    """Check if swapping x_out for x_in is valid (no 4-APs in new set)."""
    # New set is S' = S \ {x_out} U {x_in}
    # A 4-AP is formed if it contains x_in and 3 other elements from S'
    for ap_idx in _elem_to_4aps[x_in]:
        a, b, c, d = _four_aps[ap_idx]
        if x_in == a:
            others = (b, c, d)
        elif x_in == b:
            others = (a, c, d)
        elif x_in == c:
            others = (a, b, d)
        else:
            others = (a, b, c)
        
        # All others must be in S' = S \ {x_out}
        # i.e., all in S and none equal to x_out
        all_in = True
        for y in others:
            if y not in S or y == x_out:
                all_in = False
                break
        if all_in:
            return False
    return True


def swap_gain(S, x_out, x_in):
    """Net change in 3-AP count for swapping x_out out and x_in in."""
    # Loss from removing x_out
    loss = 0
    for ap_idx in _elem_to_3aps[x_out]:
        a, b, c = _three_aps[ap_idx]
        if x_out == a:
            if b in S and c in S:
                loss += 1
        elif x_out == b:
            if a in S and c in S:
                loss += 1
        else:
            if a in S and b in S:
                loss += 1
    
    # Gain from adding x_in to S \ {x_out}
    gain = 0
    for ap_idx in _elem_to_3aps[x_in]:
        a, b, c = _three_aps[ap_idx]
        if x_in == a:
            if b in S and c in S and b != x_out and c != x_out:
                gain += 1
        elif x_in == b:
            if a in S and c in S and a != x_out and c != x_out:
                gain += 1
        else:
            if a in S and b in S and a != x_out and b != x_out:
                gain += 1
    
    return gain - loss


def local_search(S, time_limit=None):
    """
    Perform local search until no improvement is found or time limit is reached.
    Uses Add and Swap moves.
    """
    S = set(S)
    current_score = count_3aps(S)
    not_in_S = set(range(1, N + 1)) - S
    improved = True
    start_time = time.time()
    
    while improved:
        if time_limit and time.time() - start_time > time_limit:
            break
            
        improved = False
        best_move = None
        best_gain = 0
        move_type = None
        move_data = None
        
        # 1. Try Add moves
        # Iterate over candidates not in S
        # Optimization: We can just iterate all x not in S
        for x in not_in_S:
            if is_valid_add(S, x):
                gain = delta_add(S, x)
                if gain > best_gain:
                    best_gain = gain
                    best_move = x
                    move_type = 'add'
        
        # 2. Try Swap moves
        # Only if no add move found or swap is better? 
        # Usually, if we can add, we should. But sometimes a swap allows a better structure later?
        # In local search, we take the best immediate improvement.
        # If best_gain from add is > 0, we take it.
        # If not, we check swaps.
        if best_gain <= 0:
            # Try swaps
            in_S_list = list(S)
            for x_out in in_S_list:
                for x_in in not_in_S:
                    if is_valid_swap(S, x_out, x_in):
                        g = swap_gain(S, x_out, x_in)
                        if g > best_gain:
                            best_gain = g
                            best_move = (x_out, x_in)
                            move_type = 'swap'
        
        if best_gain > 0:
            if move_type == 'add':
                x = best_move
                S.add(x)
                not_in_S.discard(x)
            else:
                x_out, x_in = best_move
                S.discard(x_out)
                S.add(x_in)
                not_in_S.discard(x_in)
                not_in_S.add(x_out)
            
            current_score += best_gain
            improved = True
    
    return S, current_score


def greedy_construct(seed=0):
    """
    Greedy construction: add elements in order of highest marginal 3-AP gain
    while maintaining no-4-AP constraint.
    """
    random.seed(seed)
    S = set()
    available = list(range(1, N + 1))
    
    while available:
        best_gain = -1
        best_elem = None
        candidates = []
        for x in available:
            if not is_valid_add(S, x):
                continue
            gain = delta_add(S, x)
            candidates.append((gain, x))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Pick best or from top-K for diversity
        top_k = min(5, len(candidates))
        if random.random() < 0.8:
            best_elem = candidates[0][1]
        else:
            best_elem = random.choice([c[1] for c in candidates[:top_k]])
        
        S.add(best_elem)
        available.remove(best_elem)
    
    return S


def _exp(x):
    """Safe exponent for acceptance probability."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return math.exp(x)


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Phase 1: Initial Greedy Constructions
    best_S = set()
    best_score = -1
    
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        S = greedy_construct(seed)
        S, score = local_search(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Phase 2: Iterated Local Search (ILS) with Simulated Annealing on Optima
    # We treat each local optimum as a node in a new graph.
    # Moves: Perturb -> Local Search -> Accept/Reject
    
    current_S = best_S
    current_score = best_score
    T = 5.0  # Initial temperature for SA on optima
    cooling = 0.999
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        # 1. Perturb current_S
        # Strategy: Remove k elements. k adapts to stagnation.
        k_remove = random.randint(3, 8)
        if no_improve_count > 50:
            k_remove = random.randint(8, 15)
        
        perturbed_S = set(current_S)
        elems = list(perturbed_S)
        random.shuffle(elems)
        for x in elems[:k_remove]:
            perturbed_S.discard(x)
        
        # Ensure validity (should be valid as subset of valid set)
        
        # 2. Local Search on perturbed_S
        new_S, new_score = local_search(perturbed_S)
        
        # 3. Acceptance Criterion
        delta_score = new_score - current_score
        
        if delta_score > 0:
            # Accept improvement
            current_S = new_S
            current_score = new_score
            no_improve_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_S = set(new_S)
        else:
            # Accept worse solution with probability exp(delta / T)
            if random.random() < _exp(delta_score / max(T, 0.1)):
                current_S = new_S
                current_score = new_score
                no_improve_count += 1
            else:
                # Reject, keep current
                no_improve_count += 1
        
        # Update Temperature
        T *= cooling
        if T < 0.5:
            T = 0.5 # Floor to allow some exploration
            
        # Optional: Restart if stuck for too long
        if no_improve_count > 100:
            # Try a fresh greedy restart
            S = greedy_construct(random.randint(0, 1000))
            S, score = local_search(S)
            if score > current_score:
                current_S = S
                current_score = score
                no_improve_count = 0
                T = 5.0 # Reset temperature
            else:
                no_improve_count = 0 # Reset counter to try again
                T = max(T, 2.0)

    # Final Polish
    final_S, final_score = local_search(best_S)
    if final_score > best_score:
        best_S = final_S
        best_score = final_score

    # Final verification
    if has_4ap(best_S):
        # Fallback
        best_S = greedy_construct(0)
        best_S, _ = local_search(best_S)
        
    return sorted(best_S)


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, 3-APs: {count_3aps(result)}")
    print(f"Has 4-AP: {has_4ap(result)}")
    print(f"First 20: {result[:20]}")
