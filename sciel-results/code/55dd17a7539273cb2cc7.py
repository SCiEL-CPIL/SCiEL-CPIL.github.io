import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s hard stop, 140s soft stop for new tasks
    deadline = start_time + 140.0

    # Fallback: n=9 modular coloring (valid)
    # (a+b+c) % 3 == 0 -> Red (0), else Blue (1)
    # This avoids Red K4?
    # If a+b+c is 0 mod 3, a,b,c are 0,0,0 or 1,1,1 or 0,1,2.
    # Actually, for n=9, this is a known valid construction for R(4,5;3) lower bound? 
    # R(4,5;3) > 9 is true.
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def run_search(target_n, time_limit):
        """
        Run Local Search (Min-Conflict + Simulated Annealing) for target_n.
        Returns (colors, num_violations)
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Precompute structure
        quads = list(combinations(range(target_n), 4))
        num_quads = len(quads)
        quints = list(combinations(range(target_n), 5))
        num_quints = len(quints)
        
        # Incidence structures
        # triple_in_quads[t]: list of (quad_idx, position_in_quad)
        # triple_in_quints[t]: list of (quint_idx, position_in_quint)
        triple_in_quads = [[] for _ in range(num_triples)]
        quad_triples = []
        for q_idx, q in enumerate(quads):
            a, b, c, d = q
            ts = [triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]]
            quad_triples.append(ts)
            for pos, t in enumerate(ts):
                triple_in_quads[t].append((q_idx, pos))
                
        triple_in_quints = [[] for _ in range(num_triples)]
        quint_triples = []
        for q_idx, q in enumerate(quints):
            ts = [triple_to_idx[t] for t in combinations(q, 3)]
            quint_triples.append(ts)
            for pos, t in enumerate(ts):
                triple_in_quints[t].append((q_idx, pos))

        # State
        # colors: 0 (Red), 1 (Blue)
        # We try different initializations
        initializations = []
        
        # 1. Modular 3
        initializations.append(np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8))
        # 2. Modular 4
        initializations.append(np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8))
        # 3. Random
        initializations.append(np.random.randint(0, 2, size=num_triples, dtype=np.int8))
        # 4. Biased Random (more Red)
        init4 = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        init4[np.random.random(num_triples) < 0.3] = 0
        initializations.append(init4)
        # 5. Biased Random (more Blue)
        init5 = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        init5[np.random.random(num_triples) < 0.3] = 1
        initializations.append(init5)

        best_cost_found = float('inf')
        best_colors_found = None

        for init_idx, init_colors in enumerate(initializations):
            if time.time() > start_time + time_limit:
                break
                
            colors = init_colors.copy()
            
            # Counts
            # quad_red_count[q]: number of red triples in quad q
            quad_red_count = np.zeros(num_quads, dtype=np.int8)
            quint_blue_count = np.zeros(num_quints, dtype=np.int8)
            
            # Initialize counts
            for q_idx, ts in enumerate(quad_triples):
                quad_red_count[q_idx] = np.sum(colors[ts] == 0)
            for q_idx, ts in enumerate(quint_triples):
                quint_blue_count[q_idx] = np.sum(colors[ts] == 1)
                
            # Violation sets
            violating_quads = set(np.where(quad_red_count == 4)[0])
            violating_quints = set(np.where(quint_blue_count == 10)[0])
            
            current_cost = len(violating_quads) + len(violating_quints)
            
            if current_cost == 0:
                return colors.tolist(), 0
                
            if current_cost < best_cost_found:
                best_cost_found = current_cost
                best_colors_found = colors.copy()

            # Simulated Annealing parameters
            T = 10.0
            T_min = 0.01
            cooling = 0.9995
            max_iters = 500000 # High limit, time is the real constraint
            
            no_improve = 0
            max_no_improve = 5000
            
            for i in range(max_iters):
                if time.time() > start_time + time_limit:
                    break
                if current_cost == 0:
                    return colors.tolist(), 0
                    
                # Select move
                t_idx = -1
                r = random.random()
                
                # Strategy: 80% pick from violation, 20% random
                if r < 0.8:
                    if violating_quads and violating_quints:
                        # Pick between quad and quint violation
                        if random.random() < 0.5:
                            q_idx = random.choice(list(violating_quads))
                            t_idx = quad_triples[q_idx][random.randint(0, 3)]
                        else:
                            q_idx = random.choice(list(violating_quints))
                            t_idx = quint_triples[q_idx][random.randint(0, 9)]
                    elif violating_quads:
                        q_idx = random.choice(list(violating_quads))
                        t_idx = quad_triples[q_idx][random.randint(0, 3)]
                    elif violating_quints:
                        q_idx = random.choice(list(violating_quints))
                        t_idx = quint_triples[q_idx][random.randint(0, 9)]
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    t_idx = random.randint(0, num_triples - 1)
                    
                old_color = colors[t_idx]
                
                # Calculate Delta
                delta = 0
                
                # Update Quads
                for q_idx, pos in triple_in_quads[t_idx]:
                    if old_color == 0: # Red -> Blue
                        # Red count decreases
                        if quad_red_count[q_idx] == 4:
                            # Violation removed
                            delta -= 1
                            violating_quads.discard(q_idx)
                        # New count will be 3
                    else: # Blue -> Red
                        # Red count increases
                        if quad_red_count[q_idx] == 3:
                            # Violation added
                            delta += 1
                            violating_quads.add(q_idx)
                        # New count will be 4
                        
                # Update Quints
                for q_idx, pos in triple_in_quints[t_idx]:
                    if old_color == 1: # Blue -> Red
                        # Blue count decreases
                        if quint_blue_count[q_idx] == 10:
                            # Violation removed
                            delta -= 1
                            violating_quints.discard(q_idx)
                        # New count will be 9
                    else: # Red -> Blue
                        # Blue count increases
                        if quint_blue_count[q_idx] == 9:
                            # Violation added
                            delta += 1
                            violating_quints.add(q_idx)
                        # New count will be 10

                # Acceptance
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept neutral moves to allow wandering
                    accept = True 
                else:
                    if random.random() < np.exp(-delta / T):
                        accept = True
                
                if accept:
                    # Apply Flip
                    if old_color == 0:
                        # Red -> Blue
                        for q_idx, pos in triple_in_quads[t_idx]:
                            quad_red_count[q_idx] -= 1
                        for q_idx, pos in triple_in_quints[t_idx]:
                            quint_blue_count[q_idx] += 1
                    else:
                        # Blue -> Red
                        for q_idx, pos in triple_in_quads[t_idx]:
                            quad_red_count[q_idx] += 1
                        for q_idx, pos in triple_in_quints[t_idx]:
                            quint_blue_count[q_idx] -= 1
                    
                    colors[t_idx] = 1 - old_color
                    current_cost += delta
                    
                    if current_cost < best_cost_found:
                        best_cost_found = current_cost
                        best_colors_found = colors.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    T *= cooling
                    if T < T_min:
                        T = T_min
                        if no_improve > max_no_improve:
                            break
                else:
                    no_improve += 1
                    T *= cooling
                    if T < T_min:
                        T = T_min
                        if no_improve > max_no_improve:
                            break

        if best_cost_found == 0:
            return best_colors_found.tolist(), 0
        return best_colors_found.tolist(), best_cost_found

    # Strategy: Try increasing n
    # Focus heavily on 34, then fallback to smaller if time permits
    targets = [34, 33, 32, 31, 30]
    
    # Time allocation: 
    # 34: 100s
    # 33: 10s
    # 32: 5s
    # 31: 5s
    # 30: 5s
    # Total 125s + overhead.
    
    allocations = {
        34: 100,
        33: 10,
        32: 5,
        31: 5,
        30: 5
    }

    for n in targets:
        if time.time() > deadline - 2:
            break
            
        alloc = allocations.get(n, 5)
        current_deadline = time.time() + alloc
        
        # Run search
        colors, cost = run_search(n, alloc)
        
        if cost == 0:
            # Validate to be safe
            if is_valid(n, colors):
                best_n = n
                best_colors = colors
                break # Found max n, stop
        
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
