import time
import random
import math

N = 256
PENALTY = 10000  # Large penalty for 4-APs to ensure they are avoided

# Precompute QUBO coefficients
# Objective: Maximize sum(3-AP indicators) - PENALTY * sum(4-AP indicators)
# In binary form x_i in {0,1}:
# 3-AP (a,b,c): x_a * x_b + x_b * x_c + x_a * x_c
# 4-AP (a,b,c,d): x_a*x_b*x_c + x_a*x_b*x_d + x_a*x_c*x_d + x_b*x_c*x_d (simplified penalty: just count if all 4 present)
# Actually, for QUBO we usually pair-wise. Let's use a simpler penalty:
# If any 4-AP is fully present, subtract PENALTY.
# This is a high-order term. We can linearize or just use a large penalty on the product.
# Since we are doing Tabu Search, we can just evaluate the delta of the objective function.
# Delta for adding x:
#   Gain: Number of 3-APs completed by x (pairs in A)
#   Loss: PENALTY * Number of 4-APs completed by x (triples in A)
# Delta for removing x:
#   Loss: Number of 3-APs containing x (pairs in A\{x})
#   Gain: PENALTY * Number of 4-APs containing x (triples in A\{x})

# Precompute neighbors for fast delta calculation
# For each element x, list of:
# 3-AP pairs involving x: (y, z) such that {x,y,z} is a 3-AP
# 4-AP triples involving x: (y, z, w) such that {x,y,z,w} is a 4-AP

three_ap_pairs = [[] for _ in range(N + 1)]
four_ap_triples = [[] for _ in range(N + 1)]

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        # 3-APs
        three_ap_pairs[a].append((b, c))
        three_ap_pairs[b].append((a, c))
        three_ap_pairs[c].append((a, b))

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # 4-APs
        four_ap_triples[a].append((b, c, e))
        four_ap_triples[b].append((a, c, e))
        four_ap_triples[c].append((a, b, e))
        four_ap_triples[e].append((a, b, c))

def elem_to_bit(e):
    return 1 << (e - 1)

def mask_to_list(A_mask):
    return [i for i in range(1, N + 1) if (A_mask >> (i - 1)) & 1]

def greedy_initial_mask(seed_val=0):
    rng = random.Random(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # Use a simple greedy that avoids 4-APs
    for x in candidates:
        # Check if adding x creates a 4-AP
        valid = True
        for y, z, w in four_ap_triples[x]:
            if (A_mask & elem_to_bit(y)) and (A_mask & elem_to_bit(z)) and (A_mask & elem_to_bit(w)):
                valid = False
                break
        if valid:
            A_mask |= elem_to_bit(x)
    return A_mask

def tabu_search(A_mask_init, deadline):
    A_mask = A_mask_init
    current_score = 0
    
    # Calculate initial score
    # 3-APs: count all
    for d in range(1, N // 2 + 1):
        for start in range(1, N - 2 * d + 1):
            a, b, c = start, start + d, start + 2 * d
            if (A_mask & elem_to_bit(a)) and (A_mask & elem_to_bit(b)) and (A_mask & elem_to_bit(c)):
                current_score += 1
    
    # 4-APs: count all and subtract penalty
    for d in range(1, N // 3 + 1):
        for start in range(1, N - 3 * d + 1):
            a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
            if (A_mask & elem_to_bit(a)) and (A_mask & elem_to_bit(b)) and (A_mask & elem_to_bit(c)) and (A_mask & elem_to_bit(e)):
                current_score -= PENALTY

    best_mask = A_mask
    best_score = current_score
    
    # Tabu list: (element, time_added)
    tabu_list = []
    tabu_tenure = 50
    
    # Move history for diversification
    visited = set()
    visited.add(A_mask)
    
    start_time = time.time()
    
    while time.time() < deadline:
        if time.time() - start_time > 0.1: # Check every 0.1s
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
            start_time = time.time()
            
        # Generate neighbors
        neighbors = []
        
        # Try additions
        for x in range(1, N + 1):
            if not (A_mask & elem_to_bit(x)):
                # Check tabu
                is_tabu = False
                for elem, t in tabu_list:
                    if elem == x and t > 50: # Simplified tenure check
                        is_tabu = True
                        break
                
                # Calculate delta
                delta = 0
                # 3-AP gain
                for y, z in three_ap_pairs[x]:
                    if (A_mask & elem_to_bit(y)) and (A_mask & elem_to_bit(z)):
                        delta += 1
                # 4-AP penalty
                for y, z, w in four_ap_triples[x]:
                    if (A_mask & elem_to_bit(y)) and (A_mask & elem_to_bit(z)) and (A_mask & elem_to_bit(w)):
                        delta -= PENALTA if False else PENALTY
                
                new_score = current_score + delta
                neighbors.append((new_score, x, 'add', is_tabu))
        
        # Try removals
        for x in range(1, N + 1):
            if (A_mask & elem_to_bit(x)):
                is_tabu = False
                for elem, t in tabu_list:
                    if elem == x and t > 50:
                        is_tabu = True
                        break
                
                delta = 0
                # 3-AP loss
                for y, z in three_ap_pairs[x]:
                    if (A_mask & elem_to_bit(y)) and (A_mask & elem_to_bit(z)):
                        delta -= 1
                # 4-AP gain (removing penalty)
                for y, z, w in four_ap_triples[x]:
                    if (A_mask & elem_to_bit(y)) and (A_mask & elem_to_bit(z)) and (A_mask & elem_to_bit(w)):
                        delta += PENALTY
                
                new_score = current_score + delta
                neighbors.append((new_score, x, 'remove', is_tabu))
        
        # Sort neighbors by score
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # Select best non-tabu move, or best move if all are tabu (aspiration criterion)
        best_move = None
        for score, x, move_type, is_tabu in neighbors:
            if not is_tabu:
                best_move = (score, x, move_type)
                break
            # Aspiration: if this move improves global best, allow it
            if score > best_score:
                best_move = (score, x, move_type)
                break
        
        if not best_move:
            break
            
        new_score, x, move_type = best_move
        
        # Apply move
        if move_type == 'add':
            A_mask |= elem_to_bit(x)
        else:
            A_mask &= ~elem_to_bit(x)
        
        current_score = new_score
        
        # Update tabu list
        tabu_list.append((x, 0))
        for i in range(len(tabu_list)):
            tabu_list[i] = (tabu_list[i][0], tabu_list[i][1] + 1)
        
        # Remove old tabu entries
        tabu_list = [(elem, t) for elem, t in tabu_list if t <= tabu_tenure]
        
        # Diversification: if stuck, restart
        if len(visited) > 1000:
            A_mask = greedy_initial_mask(random.randint(0, 100))
            current_score = 0
            # Recalc score
            for d in range(1, N // 2 + 1):
                for start in range(1, N - 2 * d + 1):
                    a, b, c = start, start + d, start + 2 * d
                    if (A_mask & elem_to_bit(a)) and (A_mask & elem_to_bit(b)) and (A_mask & elem_to_bit(c)):
                        current_score += 1
            for d in range(1, N // 3 + 1):
                for start in range(1, N - 3 * d + 1):
                    a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
                    if (A_mask & elem_to_bit(a)) and (A_mask & elem_to_bit(b)) and (A_mask & elem_to_bit(c)) and (A_mask & elem_to_bit(e)):
                        current_score -= PENALTY
            visited.clear()
            visited.add(A_mask)
            tabu_list.clear()
            
        visited.add(A_mask)
        
    if current_score > best_score:
        best_mask = A_mask
        
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_mask = 0
    best_score = -1
    
    # Generate multiple diverse initial solutions
    for seed in range(5):
        if time.time() > deadline - 10:
            break
            
        init_mask = greedy_initial_mask(seed)
        
        # Run tabu search
        opt_mask = tabu_search(init_mask, deadline)
        
        # Calculate final score (3-APs only, since 4-APs are penalized heavily)
        score = 0
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if (opt_mask & elem_to_bit(a)) and (opt_mask & elem_to_bit(b)) and (opt_mask & elem_to_bit(c)):
                    score += 1
        
        # Verify no 4-APs
        valid = True
        for d in range(1, N // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if (opt_mask & elem_to_bit(a)) and (opt_mask & elem_to_bit(b)) and (opt_mask & elem_to_bit(c)) and (opt_mask & elem_to_bit(e)):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and score > best_score:
            best_score = score
            best_mask = opt_mask
            
    if best_mask == 0:
        # Fallback
        best_mask = greedy_initial_mask(0)
        
    return mask_to_list(best_mask)
