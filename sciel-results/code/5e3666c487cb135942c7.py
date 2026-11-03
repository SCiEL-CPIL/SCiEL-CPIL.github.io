import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Faster scoring using numpy for larger sets, falling back to python for small.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums
    # i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences
    # i < j
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(base, m):
    """
    Construct a set U = { floor(base^k) for k=0..m }
    Ensures 0 is in the set (base^0 = 1, so we start from k=1? No, k=0 is 1.
    We need 0 in U. So U = {0} U { floor(base^k) for k=1..m }
    """
    U = [0]
    val = 1
    for _ in range(m):
        val = int(val * base)
        if val <= 0:
            break
        if val not in U:
            U.append(val)
        else:
            # If collision, just increment to ensure uniqueness? 
            # Or stop? 
            # For a "geometric" progression, if base is not integer, floor might collide.
            # e.g. base=1.5, 1, 1, 2... 
            # Let's just use unique sorted values.
            pass
    return sorted(set(U))

def ternary_search_geometric(deadline):
    """
    Search over base b and length m for geometric-like sets.
    """
    best_score = -1
    best_set = [0, 1]
    
    # Range of bases: 1.1 to 5.0
    # Range of m: 3 to 30
    
    # Coarse search
    b_low, b_high = 1.1, 5.0
    m_low, m_high = 3, 30
    
    # We can't do full 2D ternary easily without many evaluations.
    # Let's iterate over m and ternary search b for each m.
    
    for m in range(m_low, m_high + 1):
        if time.time() > deadline - 10:
            break
        
        b_low, b_high = 1.1, 5.0
        # Ternary search for optimal base for this m
        for _ in range(10):  # 10 iterations of ternary search is ~30 evaluations
            if time.time() > deadline - 5:
                break
            b1 = b_low + (b_high - b_low) / 3
            b2 = b_high - (b_high - b_low) / 3
            
            U1 = geometric_set(b1, m)
            U2 = geometric_set(b2, m)
            
            s1 = fast_score(U1)
            s2 = fast_score(U2)
            
            if s1 > s2:
                b_high = b2
            else:
                b_low = b1
                
            if s1 > best_score:
                best_score = s1
                best_set = U1
            if s2 > best_score:
                best_score = s2
                best_set = U2
                
    return best_set, best_score

def simulated_annealing(U, deadline, max_iter=2000):
    """
    Simulated annealing to improve the set U.
    Moves:
    1. Perturb an element.
    2. Add an element.
    3. Remove an element.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.995
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        move_type = random.choice(['perturb', 'add', 'remove', 'perturb'])
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Try to move within a reasonable range
            # Range depends on the scale of the set
            max_val = max(current)
            scale = max(1, int(max_val * 0.1))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue # Skip if collision caused size change
        
        elif move_type == 'add' and n < 25:
            max_val = max(current)
            # Try adding near existing elements or gaps
            if random.random() < 0.5:
                # Near max
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.2)))
            else:
                # Random in range
                new_val = random.randint(1, max_val * 2)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue # Duplicate
        
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score < 0:
            # Invalid move
            if random.random() < T:
                current = new_set
                current_score = new_score
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current
            else:
                # Accept worse move with probability
                delta = current_score - new_score
                if random.random() < np.exp(-delta / T):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001 # Floor for temperature to allow some exploration
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (limit is 150s, evaluator kills at 150s, but we want to return well before)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # 1. Geometric Search
    # The geometric structure often provides good sumset/difference-set properties.
    # We search for the best geometric progression parameters.
    geo_set, geo_score = ternary_search_geometric(deadline)
    if geo_score > best_score:
        best_score = geo_score
        best_set = geo_set
        
    # 2. Add known good candidates (Golomb Rulers, etc.)
    candidates = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        geo_set
    ]
    
    # Generate some random sparse sets
    random.seed(42)
    for _ in range(10):
        n_elem = random.randint(4, 12)
        max_val = random.randint(50, 200)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Geometric sets with specific ratios
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(10):
            U.append(val)
            val = int(val * ratio)
            if val > 1000:
                break
        candidates.append(U)
        
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
            
    scored_candidates.sort(reverse=True)
    
    # 3. Run SA on top candidates
    # Prioritize the geometric best and the global best
    for score, U in scored_candidates[:10]:
        if time.time() > deadline - 10:
            break
        refined, refined_score = simulated_annealing(U, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # Final polish on the best set
    if time.time() < deadline - 5:
        refined, refined_score = simulated_annealing(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
