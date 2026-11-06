"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def make_triple_index(n):
        triples = list(combinations(range(n), 3))
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx
    
    def check_valid(n, colors):
        triples, tidx = make_triple_index(n)
        for quad in combinations(range(n), 4):
            count_red = 0
            for tri in combinations(quad, 3):
                if colors[tidx[tri]] == 0:
                    count_red += 1
            if count_red == 4:
                return False
        for quint in combinations(range(n), 5):
            count_blue = 0
            for tri in combinations(quint, 3):
                if colors[tidx[tri]] == 1:
                    count_blue += 1
            if count_blue == 10:
                return False
        return True
    
    def try_structured(n):
        """Try quick structured colorings for small n."""
        triples, _ = make_triple_index(n)
        # Modular coloring
        for m in range(2, min(20, n + 5)):
            for threshold in range(1, m):
                colors = [0 if (a + b + c) % m < threshold else 1 for a, b, c in triples]
                if check_valid(n, colors):
                    return colors
        # XOR-based
        for bits in range(1, 6):
            colors = [0 if (a ^ b ^ c) < (1 << bits) else 1 for a, b, c in triples]
            if check_valid(n, colors):
                return colors
        return None
    
    def simulated_annealing(n, max_time):
        """Use simulated annealing to find a valid coloring for n vertices."""
        triples, tidx = make_triple_index(n)
        num_triples = len(triples)
        
        sets_4 = list(combinations(range(n), 4))
        sets_5 = list(combinations(range(n), 5))
        num_4sets = len(sets_4)
        num_5sets = len(sets_5)
        
        # Build inverted indices
        edge_to_4sets = [[] for _ in range(num_triples)]
        edge_to_5sets = [[] for _ in range(num_triples)]
        
        for i, s4 in enumerate(sets_4):
            t1 = (s4[0], s4[1], s4[2])
            t2 = (s4[0], s4[1], s4[3])
            t3 = (s4[0], s4[2], s4[3])
            t4 = (s4[1], s4[2], s4[3])
            for t in (t1, t2, t3, t4):
                edge_to_4sets[tidx[t]].append(i)
        
        for i, s5 in enumerate(sets_5):
            for t in combinations(s5, 3):
                edge_to_5sets[tidx[t]].append(i)
        
        best_energy = float('inf')
        best_colors = None
        
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            colors = np.random.randint(0, 2, num_triples)
            
            red_counts_4 = np.zeros(num_4sets, dtype=np.int32)
            blue_counts_5 = np.zeros(num_5sets, dtype=np.int32)
            
            for i, c in enumerate(colors):
                if c == 0:
                    for s4_idx in edge_to_4sets[i]:
                        red_counts_4[s4_idx] += 1
                else:
                    for s5_idx in edge_to_5sets[i]:
                        blue_counts_5[s5_idx] += 1
            
            current_energy = np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10)
            
            if current_energy == 0:
                return colors.tolist()
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors.copy()
            
            T = 2.0
            T_min = 0.05
            cooling_rate = 0.9995
            
            while time.time() < deadline - 5:
                edge_idx = np.random.randint(num_triples)
                old_color = colors[edge_idx]
                new_color = 1 - old_color
                
                delta_energy = 0
                
                if old_color == 0:  # Red -> Blue
                    for s4_idx in edge_to_4sets[edge_idx]:
                        old_c = red_counts_4[s4_idx]
                        red_counts_4[s4_idx] = old_c - 1
                        if old_c == 4:
                            delta_energy -= 1
                    for s5_idx in edge_to_5sets[edge_idx]:
                        old_c = blue_counts_5[s5_idx]
                        blue_counts_5[s5_idx] = old_c + 1
                        if old_c == 9:
                            delta_energy += 1
                else:  # Blue -> Red
                    for s4_idx in edge_to_4sets[edge_idx]:
                        old_c = red_counts_4[s4_idx]
                        red_counts_4[s4_idx] = old_c + 1
                        if old_c == 3:
                            delta_energy += 1
                    for s5_idx in edge_to_5sets[edge_idx]:
                        old_c = blue_counts_5[s5_idx]
                        blue_counts_5[s5_idx] = old_c - 1
                        if old_c == 10:
                            delta_energy -= 1
                
                if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / T):
                    colors[edge_idx] = new_color
                    current_energy += delta_energy
                    if current_energy == 0:
                        return colors.tolist()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors.copy()
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min
        
        if best_colors is not None and best_energy == 0:
            return best_colors.tolist()
        return None
    
    # Try structured colorings for small n first (quick)
    for n in range(15, 10, -1):
        if time.time() > deadline:
            break
        colors = try_structured(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Try simulated annealing for larger n
    for n in range(34, 15, -1):
        if time.time() > deadline - 10:
            break
        colors = simulated_annealing(n, deadline - time.time())
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Fallback: guaranteed valid small coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
