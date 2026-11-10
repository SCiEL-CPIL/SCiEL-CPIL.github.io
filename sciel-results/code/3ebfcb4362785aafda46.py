"""
QUBO-style simulated annealing with O(1) energy deltas for 3-AP maximization
under 4-AP-free constraint.
"""
import time
import random

N = 256

def construct():
    start = time.time()
    deadline = start + 90 * 0.85  # 76.5s, safety margin

    # Precompute all 3-APs as tuples (a, b, c) with a < b < c
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # Precompute all 4-APs as tuples (a, b, c, e) with a < b < c < e
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element x, precompute:
    # - ap3_partners[x]: list of (y, z) such that {x,y,z} forms a 3-AP
    # - ap4_conflicts[x]: list of (y, z, w) such that {x,y,z,w} forms a 4-AP
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_conflicts = [[] for _ in range(N + 1)]
    
    for (a, b, c) in three_aps:
        # a is first: partners b, c
        ap3_partners[a].append((b, c))
        # b is middle: partners a, c
        ap3_partners[b].append((a, c))
        # c is last: partners a, b
        ap3_partners[c].append((a, b))
    
    for (a, b, c, e) in four_aps:
        # a is first: conflicts b, c, e
        ap4_conflicts[a].append((b, c, e))
        # b is second: conflicts a, c, e
        ap4_conflicts[b].append((a, c, e))
        # c is third: conflicts a, b, e
        ap4_conflicts[c].append((a, b, e))
        # e is fourth: conflicts a, b, c
        ap4_conflicts[e].append((a, b, c))

    def count_3aps(S):
        """Count 3-APs in set S."""
        count = 0
        for (a, b, c) in three_aps:
            if a in S and b in S and c in S:
                count += 1
        return count

    def has_4ap(S):
        """Check if S contains any 4-AP."""
        for (a, b, c, e) in four_aps:
            if a in S and b in S and c in S and e in S:
                return True
        return False

    def greedy_init(seed):
        """Greedy construction with randomization."""
        rng = random.Random(seed)
        S = set()
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            # Check if adding x creates a 4-AP
            valid = True
            for (y, z, w) in ap4_conflicts[x]:
                if y in S and z in S and w in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S

    def simulate_annealing(S, deadline, seed):
        """Simulated annealing to maximize 3-APs while maintaining 4-AP-free."""
        rng = random.Random(seed)
        S = set(S)
        current_score = count_3aps(S)
        best_score = current_score
        best_S = set(S)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.9995
        
        T = T_start
        step = 0
        
        while time.time() < deadline and T > T_end:
            step += 1
            
            # Choose a random move: add, remove, or swap
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Try to add a random element not in S
                candidates = [x for x in range(1, N + 1) if x not in S]
                if not candidates:
                    T *= cooling_rate
                    continue
                x = rng.choice(candidates)
                
                # Check if adding x creates a 4-AP
                valid = True
                for (y, z, w) in ap4_conflicts[x]:
                    if y in S and z in S and w in S:
                        valid = False
                        break
                if not valid:
                    T *= cooling_rate
                    continue
                
                # Calculate gain in 3-APs
                gain = 0
                for (y, z) in ap3_partners[x]:
                    if y in S and z in S:
                        gain += 1
                
                if gain > 0:
                    S.add(x)
                    current_score += gain
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                else:
                    # Accept with probability exp(gain/T) if gain < 0
                    if rng.random() < 0:  # gain is 0, no benefit
                        pass
                T *= cooling_rate
                
            elif move_type == 'remove':
                # Try to remove a random element from S
                if not S:
                    T *= cooling_rate
                    continue
                x = rng.choice(list(S))
                
                # Calculate loss in 3-APs
                loss = 0
                for (y, z) in ap3_partners[x]:
                    if y in S and z in S:
                        loss += 1
                
                delta = -loss
                # Always accept removal if it helps (delta > 0) or with probability
                if delta > 0 or (T > 0 and rng.random() < min(1.0, delta / T)):
                    S.discard(x)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                T *= cooling_rate
                
            else:  # swap
                # Try to swap: remove x from S, add y not in S
                if not S:
                    T *= cooling_rate
                    continue
                x = rng.choice(list(S))
                candidates = [y for y in range(1, N + 1) if y not in S]
                if not candidates:
                    T *= cooling_rate
                    continue
                y = rng.choice(candidates)
                
                # Check if y can be added to S \ {x} without creating 4-AP
                S_temp = S - {x}
                valid = True
                for (a, b, c) in ap4_conflicts[y]:
                    if a in S_temp and b in S_temp and c in S_temp:
                        valid = False
                        break
                if not valid:
                    T *= cooling_rate
                    continue
                
                # Calculate net change in 3-APs
                # Loss from removing x
                loss_x = 0
                for (a, b) in ap3_partners[x]:
                    if a in S and b in S:
                        loss_x += 1
                
                # Gain from adding y to S \ {x}
                gain_y = 0
                for (a, b) in ap3_partners[y]:
                    if a in S_temp and b in S_temp:
                        gain_y += 1
                
                # Subtract shared 3-APs (those containing both x and y)
                shared = 0
                # x, y, z where z = 2y - x
                z1 = 2 * y - x
                if z1 in S and z1 != x and z1 != y:
                    shared += 1
                # y, x, z where z = 2x - y
                z2 = 2 * x - y
                if z2 in S and z2 != x and z2 != y:
                    shared += 1
                # z, x, y where z = 2x - y (same as above)
                # x, z, y where z = (x+y)/2
                if (x + y) % 2 == 0:
                    z3 = (x + y) // 2
                    if z3 in S and z3 != x and z3 != y:
                        shared += 1
                # z, y, x where z = 2y - x (same as first)
                
                gain_y_adj = gain_y - shared
                delta = gain_y_adj - loss_x
                
                if delta > 0 or (T > 0 and rng.random() < min(1.0, delta / T)):
                    S.discard(x)
                    S.add(y)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                T *= cooling_rate
            
            # Periodically cool faster if stuck
            if step % 1000 == 0:
                T *= 0.95
        
        return best_S, best_score

    # Run multiple restarts with different seeds
    best_S = set()
    best_score = -1
    
    # First, try greedy initialization with several seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        S = greedy_init(seed)
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Then apply simulated annealing to the best solution
    if best_S:
        best_S, best_score = simulate_annealing(best_S, deadline, seed=42)
    
    # Convert to sorted list
    result = sorted(best_S)
    
    if not result:
        return [1]
    
    return result
