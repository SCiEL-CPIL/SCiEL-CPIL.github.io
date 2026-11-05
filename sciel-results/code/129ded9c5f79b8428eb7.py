import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58  # Safety margin before 60s
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets using bitmasks for speed
    # adj[i] is a bitmask where bit j is set if i and j conflict
    adj = [0] * N
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden_mask = 0
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden_mask |= (1 << i)
            continue
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i] |= (1 << j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if not (forbidden_mask & (1 << i))]
    
    def is_valid_addition(v, S_mask):
        """Check if v can be added to S_mask."""
        if forbidden_mask & (1 << v):
            return False
        return (adj[v] & S_mask) == 0

    def greedy_solve(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        order = candidates[:]
        random.shuffle(order)
        
        S_mask = 0
        for v in order:
            if is_valid_addition(v, S_mask):
                S_mask |= (1 << v)
                
        return S_mask

    def greedy_by_degree(random_seed=None):
        """Greedy that picks vertex with fewest conflicts among remaining."""
        if random_seed is not None:
            random.seed(random_seed)
        
        remaining = set(candidates)
        S_mask = 0
        
        while remaining and (time.time() - start_time < DEADLINE - 2):
            # Pick vertex with minimum degree within remaining
            best_v = None
            best_deg = float('inf')
            best_conflicts = 0
            
            for v in remaining:
                # Count conflicts with other remaining vertices
                conflicts = 0
                adj_mask = adj[v]
                for u in remaining:
                    if u != v and (adj_mask >> u) & 1:
                        conflicts += 1
                
                if conflicts < best_deg or (conflicts == best_deg and random.random() < 0.5):
                    best_deg = conflicts
                    best_v = v
                    best_conflicts = conflicts
            
            if best_v is None:
                break
            
            # Add best_v if valid
            if is_valid_addition(best_v, S_mask):
                S_mask |= (1 << best_v)
                remaining.remove(best_v)
                # Remove all neighbors of best_v from remaining
                for u in list(remaining):
                    if (adj[best_v] >> u) & 1:
                        remaining.remove(u)
        
        return S_mask

    def modular_seed(modulus):
        """
        Generate a seed based on modular arithmetic.
        """
        squares_mod = set()
        for k in range(1, int(math.isqrt(modulus**2)) + 1):
            squares_mod.add((k * k) % modulus)
        
        # Find safe residues
        safe_residues = []
        for r in range(modulus):
            if (2 * r) % modulus not in squares_mod:
                safe_residues.append(r)
        
        # Find max independent set of residues
        best_size = 0
        best_residues = None
        
        # Try single residues
        for r in safe_residues:
            count = 0
            for v in candidates:
                if (v + 1) % modulus == r:
                    count += 1
            if count > best_size:
                best_size = count
                best_residues = [r]
        
        # Try pairs of residues
        for i in range(len(safe_residues)):
            for j in range(i, len(safe_residues)):
                r1 = safe_residues[i]
                r2 = safe_residues[j]
                if (r1 + r2) % modulus in squares_mod:
                    continue
                count = 0
                for v in candidates:
                    val = (v + 1) % modulus
                    if val == r1 or val == r2:
                        count += 1
                if count > best_size:
                    best_size = count
                    best_residues = [r1, r2]
        
        # Try triples of residues
        for i in range(len(safe_residues)):
            for j in range(i, len(safe_residues)):
                for k in range(j, len(safe_residues)):
                    r1, r2, r3 = safe_residues[i], safe_residues[j], safe_residues[k]
                    if (r1 + r2) % modulus in squares_mod:
                        continue
                    if (r1 + r3) % modulus in squares_mod:
                        continue
                    if (r2 + r3) % modulus in squares_mod:
                        continue
                    count = 0
                    for v in candidates:
                        val = (v + 1) % modulus
                        if val in (r1, r2, r3):
                            count += 1
                    if count > best_size:
                        best_size = count
                        best_residues = [r1, r2, r3]
        
        if best_residues is None:
            return 0
            
        S_mask = 0
        for v in candidates:
            val = (v + 1) % modulus
            if val in best_residues:
                S_mask |= (1 << v)
        
        return S_mask

    def local_improve(S_mask):
        """Try to add vertices that don't conflict."""
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if (S_mask & (1 << v)):
                    continue
                if is_valid_addition(v, S_mask):
                    S_mask |= (1 << v)
                    improved = True
        return S_mask

    def local_search_1_2(S_mask):
        """
        1-2 Exchange Local Search.
        Remove one vertex v, try to add two vertices u, w.
        Net gain: +1.
        """
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            S_list = [i for i in range(N) if S_mask & (1 << i)]
            random.shuffle(S_list)
            
            for v in S_list:
                # Remove v
                S_temp = S_mask & ~(1 << v)
                
                # Find candidates that can be added to S_temp
                available = []
                for u in candidates:
                    if (S_temp & (1 << u)):
                        continue
                    if (adj[u] & S_temp) == 0:
                        available.append(u)
                
                if len(available) < 2:
                    continue
                
                # Find an independent pair in available
                found_pair = False
                for i in range(len(available)):
                    u = available[i]
                    for j in range(i + 1, len(available)):
                        w = available[j]
                        if (adj[u] & (1 << w)) == 0:
                            S_mask = S_temp | (1 << u) | (1 << w)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    break
        
        return S_mask

    def local_search_2_1(S_mask):
        """
        2-1 Exchange Local Search.
        Remove two vertices v1, v2, try to add one vertex u.
        Net gain: -1, but may unlock larger improvements.
        Used as a diversification move.
        """
        improved = False
        if time.time() - start_time > DEADLINE - 3:
            return S_mask, improved
        
        S_list = [i for i in range(N) if S_mask & (1 << i)]
        random.shuffle(S_list)
        
        # Try removing pairs
        for i in range(min(len(S_list), 20)):
            for j in range(i + 1, min(len(S_list), 20)):
                v1, v2 = S_list[i], S_list[j]
                S_temp = S_mask & ~(1 << v1) & ~(1 << v2)
                
                # Find best single vertex to add
                best_u = None
                best_score = -1
                for u in candidates:
                    if (S_temp & (1 << u)):
                        continue
                    if (adj[u] & S_temp) == 0:
                        # Score: how many new vertices does this unlock?
                        # Simple heuristic: just pick one
                        best_u = u
                        break
                
                if best_u is not None:
                    S_new = S_temp | (1 << best_u)
                    # This is a diversification move, accept it sometimes
                    if random.random() < 0.3:
                        return S_new, True
        
        return S_mask, improved

    def simulated_annealing_polish(S_mask, max_iters=500):
        """
        Simple SA-style polish: accept worse moves with probability based on "temperature".
        """
        current = S_mask
        current_size = bin(current).count('1')
        best = current
        best_size = current_size
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 2:
                break
            
            # Pick a random move
            move_type = random.random()
            if move_type < 0.4:
                # Try to add a vertex
                S_list = [i for i in candidates if not (current & (1 << i))]
                if not S_list:
                    continue
                v = random.choice(S_list)
                if is_valid_addition(v, current):
                    new_mask = current | (1 << v)
                    new_size = bin(new_mask).count('1')
                    delta = new_size - current_size
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
            elif move_type < 0.8:
                # 1-2 exchange
                S_in = [i for i in range(N) if current & (1 << i)]
                if not S_in:
                    continue
                v = random.choice(S_in)
                S_temp = current & ~(1 << v)
                available = [u for u in candidates if not (S_temp & (1 << u)) and (adj[u] & S_temp) == 0]
                if len(available) >= 2:
                    random.shuffle(available)
                    for i in range(len(available)):
                        for j in range(i + 1, len(available)):
                            if (adj[available[i]] & (1 << available[j])) == 0:
                                new_mask = S_temp | (1 << available[i]) | (1 << available[j])
                                new_size = bin(new_mask).count('1')
                                delta = new_size - current_size
                                if delta > 0 or random.random() < math.exp(delta / T):
                                    current = new_mask
                                    current_size = new_size
                                    if current_size > best_size:
                                        best = current
                                        best_size = new_size
                                break
                        else:
                            continue
                        break
            else:
                # 2-1 exchange (diversification)
                S_in = [i for i in range(N) if current & (1 << i)]
                if len(S_in) >= 2:
                    random.shuffle(S_in)
                    v1, v2 = S_in[0], S_in[1]
                    S_temp = current & ~(1 << v1) & ~(1 << v2)
                    available = [u for u in candidates if not (S_temp & (1 << u)) and (adj[u] & S_temp) == 0]
                    if available:
                        u = random.choice(available)
                        new_mask = S_temp | (1 << u)
                        new_size = bin(new_mask).count('1')
                        delta = new_size - current_size
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current = new_mask
                            current_size = new_size
                            if current_size > best_size:
                                best = current
                                best_size = new_size
            
            T *= cooling
        
        return best

    best_size = 0
    best_mask = 0
    
    # Initial seeds
    seeds = []
    
    # Modular seeds with more moduli
    for mod in [4, 8, 16, 32, 64]:
        seeds.append(modular_seed(mod))
        
    # Random greedy seeds
    for _ in range(15):
        seeds.append(greedy_solve())
    
    # Degree-based greedy seeds
    for _ in range(5):
        seeds.append(greedy_by_degree())
    
    # Iterate
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
            
        S_mask = local_improve(seed)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
            
        # Local search
        S_mask = local_search_1_2(S_mask)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
        
        # SA polish
        S_mask = simulated_annealing_polish(S_mask, max_iters=200)
        
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask

        # Random restarts if time permits
        while time.time() - start_time < DEADLINE - 5:
            S_new = greedy_solve()
            S_new = local_improve(S_new)
            S_new = local_search_1_2(S_new)
            S_new = simulated_annealing_polish(S_new, max_iters=100)
            
            if bin(S_new).count('1') > best_size:
                best_size = bin(S_new).count('1')
                best_mask = S_new

    # Final polish on best solution
    if time.time() - start_time < DEADLINE - 3:
        best_mask = local_improve(best_mask)
        best_mask = local_search_1_2(best_mask)
        best_mask = simulated_annealing_polish(best_mask, max_iters=300)
        best_size = bin(best_mask).count('1')

    # Convert bitmask back to list of 1-based values
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
