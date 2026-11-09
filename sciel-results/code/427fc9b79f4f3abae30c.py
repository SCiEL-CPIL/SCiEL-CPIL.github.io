import random
import time
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Deterministic Tabu Search with bitset optimization.
    """
    target_n = 34
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead

    # Pre-compute triple indices for n=34
    # We will handle n dynamically in the search, but we target 34.
    
    def build_masks(n, colors):
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        triple_indices = {}
        
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_indices[(a, b, c)] = idx
                    if colors[idx] == 0:
                        red_mask[a][b] |= (1 << c)
                        red_mask[b][a] |= (1 << c)
                        red_mask[a][c] |= (1 << b)
                        red_mask[c][a] |= (1 << b)
                        red_mask[b][c] |= (1 << a)
                        red_mask[c][b] |= (1 << a)
                    else:
                        blue_mask[a][b] |= (1 << c)
                        blue_mask[b][a] |= (1 << c)
                        blue_mask[a][c] |= (1 << b)
                        blue_mask[c][a] |= (1 << b)
                        blue_mask[b][c] |= (1 << a)
                        blue_mask[c][b] |= (1 << a)
                    idx += 1
        return red_mask, blue_mask, triple_indices

    def find_violation(n, red_mask, blue_mask):
        # Find a Red K4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    if candidates:
                        lsb = candidates & -candidates
                        d = lsb.bit_length() - 1
                        return ('red', (a, b, c, d))
        
        # Find a Blue K5
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    cands = []
                    temp = candidates
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cands.append(v)
                        temp ^= lsb
                    for i in range(len(cands)):
                        d = cands[i]
                        for j in range(i+1, len(cands)):
                            e = cands[j]
                            if (blue_mask[a][d] & (1 << e)) and \
                               (blue_mask[b][d] & (1 << e)) and \
                               (blue_mask[c][d] & (1 << e)):
                                return ('blue', (a, b, c, d, e))
        return None

    def run_tabu_search(n, initial_colors, rng):
        colors = initial_colors[:]
        red_mask, blue_mask, triple_indices = build_masks(n, colors)
        
        tabu_list = {} # index -> expiration step
        tabu_size = 100
        max_steps = 50000
        no_improve_counter = 0
        best_violation_count = float('inf') # We don't easily count, so we just track if we found 0
        
        # We will just run until we find 0 violations or time out
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            violation = find_violation(n, red_mask, blue_mask)
            if violation is None:
                return colors
            
            color_type, vertices = violation
            if color_type == 'red':
                a, b, c, d = vertices
                possible_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            else:
                a, b, c, d, e = vertices
                possible_triples = list(combinations((a,b,c,d,e), 3))
            
            # Select a triple to flip
            # Deterministic choice based on step and rng
            valid_moves = []
            for t in possible_triples:
                idx = triple_indices[t]
                # Check tabu
                if idx in tabu_list and tabu_list[idx] > step:
                    continue
                valid_moves.append((t, idx))
                
            if not valid_moves:
                # If all are tabu, pick the one with oldest tabu expiration
                min_exp = float('inf')
                best_move = None
                for t in possible_triples:
                    idx = triple_indices[t]
                    exp = tabu_list.get(idx, 0)
                    if exp < min_exp:
                        min_exp = exp
                        best_move = (t, idx)
                if best_move:
                    t, idx = best_move
                else:
                    t, idx = possible_triples[0], triple_indices[possible_triples[0]]
            else:
                # Pick randomly from valid moves using deterministic rng
                t, idx = rng.choice(valid_moves)
                
            # Flip color
            old_color = colors[idx]
            new_color = 1 - old_color
            colors[idx] = new_color
            
            # Update masks
            a, b, c = t
            if old_color == 0:
                red_mask[a][b] &= ~(1 << c)
                red_mask[b][a] &= ~(1 << c)
                red_mask[a][c] &= ~(1 << b)
                red_mask[c][a] &= ~(1 << b)
                red_mask[b][c] &= ~(1 << a)
                red_mask[c][b] &= ~(1 << a)
                
                blue_mask[a][b] |= (1 << c)
                blue_mask[b][a] |= (1 << c)
                blue_mask[a][c] |= (1 << b)
                blue_mask[c][a] |= (1 << b)
                blue_mask[b][c] |= (1 << a)
                blue_mask[c][b] |= (1 << a)
            else:
                blue_mask[a][b] &= ~(1 << c)
                blue_mask[b][a] &= ~(1 << c)
                blue_mask[a][c] &= ~(1 << b)
                blue_mask[c][a] &= ~(1 << b)
                blue_mask[b][c] &= ~(1 << a)
                blue_mask[c][b] &= ~(1 << a)
                
                red_mask[a][b] |= (1 << c)
                red_mask[b][a] |= (1 << c)
                red_mask[a][c] |= (1 << b)
                red_mask[c][a] |= (1 << b)
                red_mask[b][c] |= (1 << a)
                red_mask[c][b] |= (1 << a)
                
            # Update tabu
            tabu_list[idx] = step + tabu_size

        return None

    # Try n=34
    n = target_n
    num_triples = n * (n-1) * (n-2) // 6
    
    # Define several deterministic initial states
    initial_states = []
    
    # 1. All 0
    initial_states.append([0] * num_triples)
    
    # 2. All 1
    initial_states.append([1] * num_triples)
    
    # 3. Parity (a+b+c) % 2
    parity_colors = []
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                parity_colors.append((a+b+c) % 2)
                idx += 1
    initial_states.append(parity_colors)
    
    # 4. Mod 3
    mod3_colors = []
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mod3_colors.append(0 if (a+b+c) % 3 == 0 else 1)
                idx += 1
    initial_states.append(mod3_colors)
    
    # 5. Random with fixed seed
    rng_seed = random.Random(42)
    rand_colors = [rng_seed.randint(0, 1) for _ in range(num_triples)]
    initial_states.append(rand_colors)
    
    # Run Tabu Search on each
    for i, init in enumerate(initial_states):
        if time.time() - start_time > time_limit:
            break
        rng = random.Random(i + 100) # Different seed for each restart
        result = run_tabu_search(n, init, rng)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: try smaller n? 
    # If 34 fails, 33 might work. But 34 is the known bound.
    # We will just return a small valid one to be safe, though 34 is expected.
    # Actually, if 34 fails, the score drops. 
    # Let's try n=33 if 34 fails.
    
    for n in range(33, 9, -1):
        if time.time() - start_time > time_limit:
            break
        num_triples = n * (n-1) * (n-2) // 6
        # Quick random search for smaller n
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            red_mask, blue_mask, triple_indices = build_masks(n, colors)
            # Simple local search
            for _ in range(10000):
                if time.time() - start_time > time_limit:
                    break
                violation = find_violation(n, red_mask, blue_mask)
                if violation is None:
                    return {'n': n, 'colors': colors}
                color_type, vertices = violation
                if color_type == 'red':
                    a, b, c, d = vertices
                    possible_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                else:
                    a, b, c, d, e = vertices
                    possible_triples = list(combinations((a,b,c,d,e), 3))
                t = random.choice(possible_triples)
                idx = triple_indices[t]
                old_color = colors[idx]
                new_color = 1 - old_color
                colors[idx] = new_color
                a, b, c = t
                if old_color == 0:
                    red_mask[a][b] &= ~(1 << c)
                    red_mask[b][a] &= ~(1 << c)
                    red_mask[a][c] &= ~(1 << b)
                    red_mask[c][a] &= ~(1 << b)
                    red_mask[b][c] &= ~(1 << a)
                    red_mask[c][b] &= ~(1 << a)
                    blue_mask[a][b] |= (1 << c)
                    blue_mask[b][a] |= (1 << c)
                    blue_mask[a][c] |= (1 << b)
                    blue_mask[c][a] |= (1 << b)
                    blue_mask[b][c] |= (1 << a)
                    blue_mask[c][b] |= (1 << a)
                else:
                    blue_mask[a][b] &= ~(1 << c)
                    blue_mask[b][a] &= ~(1 << c)
                    blue_mask[a][c] &= ~(1 << b)
                    blue_mask[c][a] &= ~(1 << b)
                    blue_mask[b][c] &= ~(1 << a)
                    blue_mask[c][b] &= ~(1 << a)
                    red_mask[a][b] |= (1 << c)
                    red_mask[b][a] |= (1 << c)
                    red_mask[a][c] |= (1 << b)
                    red_mask[c][a] |= (1 << b)
                    red_mask[b][c] |= (1 << a)
                    red_mask[c][b] |= (1 << a)
                    
    # Final fallback n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
