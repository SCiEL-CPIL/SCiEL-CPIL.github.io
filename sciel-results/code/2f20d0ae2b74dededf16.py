import time
import random
import math
from itertools import combinations

def construct():
    """
    Try to find the largest n with a valid 3-uniform hypergraph coloring
    where no 4 vertices have all 4 triples red, and no 5 vertices have all 10 triples blue.
    Uses algebraic constructions and local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try known constructions first
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 30:
            break
        
        # Try algebraic construction based on cyclic group Z_p
        result = try_algebraic(n)
        if result is not None:
            return {'n': n, 'colors': result}
        
        # Try improved local search with better initialization
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 5:
            result = solve_improved(n, min(time_remaining, 20))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback: try smaller n with more time
    for n in range(29, 8, -1):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 2:
            result = solve_improved(n, min(time_remaining, 10))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Guaranteed fallback for small n
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

def try_algebraic(n):
    """
    Try algebraic constructions based on properties of cyclic groups.
    For R(4,5;3), there are known constructions using finite fields or cyclic groups.
    """
    if n <= 1:
        return [0] * (1 if n == 1 else 0)
    
    # Try construction based on quadratic residues or similar
    # This is a heuristic - try several different algebraic rules
    
    edges = list(combinations(range(n), 3))
    
    # Strategy 1: Based on sum modulo p
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        colors = []
        for a, b, c in edges:
            s = (a + b + c) % mod
            # Color red (0) if s is in a "good" set
            # Try different thresholds
            if s < mod // 2:
                colors.append(0)
            else:
                colors.append(1)
        
        if is_valid(n, colors):
            return colors
    
    # Strategy 2: Based on product modulo p
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        colors = []
        for a, b, c in edges:
            p = (a * b * c) % mod
            if p < mod // 2:
                colors.append(0)
            else:
                colors.append(1)
        
        if is_valid(n, colors):
            return colors
    
    # Strategy 3: Based on differences
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        colors = []
        for a, b, c in edges:
            d1 = (b - a) % mod
            d2 = (c - b) % mod
            s = (d1 + d2) % mod
            if s < mod // 2:
                colors.append(0)
            else:
                colors.append(1)
        
        if is_valid(n, colors):
            return colors
    
    # Strategy 4: Use a fixed pattern based on position in lexicographic order
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        colors = []
        for i, (a, b, c) in enumerate(edges):
            colors.append(0 if i % period < period // 2 else 1)
        
        if is_valid(n, colors):
            return colors
    
    # Strategy 5: Weighted combination
    for w1 in range(1, 4):
        for w2 in range(1, 4):
            for mod in [5, 7, 11, 13]:
                colors = []
                for a, b, c in edges:
                    score = (w1 * (a + b + c) + w2 * (a * b * c)) % mod
                    colors.append(0 if score < mod // 2 else 1)
                
                if is_valid(n, colors):
                    return colors
    
    return None

def is_valid(n, colors):
    """Check if a coloring is valid (no red K4^(3), no blue K5^(3))"""
    if len(colors) != len(list(combinations(range(n), 3))):
        return False
    
    edge_index = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                edge_index[(a,b,c)] = idx
                idx += 1
    
    # Check for red K4: all 4 triples must be color 0
    for k4 in combinations(range(n), 4):
        a, b, c, d = k4
        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        if all(colors[edge_index[t]] == 0 for t in triples):
            return False
    
    # Check for blue K5: all 10 triples must be color 1
    for k5 in combinations(range(n), 5):
        for triple in combinations(k5, 3):
            if colors[edge_index[triple]] != 1:
                break
        else:
            # All 10 triples are blue
            return False
    
    return True

def solve_improved(n, time_limit=10):
    """
    Improved local search with better initialization and more aggressive optimization.
    """
    start_time = time.time()
    
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Precompute which edges belong to which K4 and K5
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        es = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(es)
        for e in es:
            edge_to_k5s[e].append(len(k5_edges) - 1)
    
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # Try multiple random restarts
    best_result = None
    best_violations = float('inf')
    
    num_restarts = min(5, max(1, int(time_limit / 2)))
    
    for restart in range(num_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # Better initialization: start with a biased random coloring
        # Slightly favor blue to avoid red K4s
        p_red = 0.4  # 40% red, 60% blue
        colors = [1 if random.random() > p_red else 0 for _ in range(num_edges)]
        
        # Calculate initial counts
        red_counts = [0] * num_k4s
        blue_counts = [0] * num_k5s
        
        for i, e_list in enumerate(k4_edges):
            for e in e_list:
                if colors[e] == 0:
                    red_counts[i] += 1
                    
        for i, e_list in enumerate(k5_edges):
            for e in e_list:
                if colors[e] == 1:
                    blue_counts[i] += 1
                    
        total_violations = sum(1 for rc in red_counts if rc == 4) + sum(1 for bc in blue_counts if bc == 10)
        
        if total_violations == 0:
            return colors
        
        if total_violations < best_violations:
            best_violations = total_violations
            best_result = colors.copy()
        
        # Simulated annealing with faster cooling
        T = 2.0
        cooling = 0.995
        exp = math.exp
        
        time_per_restart = (start_time + time_limit - time.time()) / (num_restarts - restart)
        
        while time.time() < start_time + time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Update K4s
            for k4_idx in edge_to_k4s[e_idx]:
                if old_color == 0:
                    red_counts[k4_idx] -= 1
                    if red_counts[k4_idx] == 3:
                        delta_v -= 1
                else:
                    red_counts[k4_idx] += 1
                    if red_counts[k4_idx] == 4:
                        delta_v += 1
                        
            # Update K5s
            for k5_idx in edge_to_k5s[e_idx]:
                if old_color == 1:
                    blue_counts[k5_idx] -= 1
                    if blue_counts[k5_idx] == 9:
                        delta_v -= 1
                else:
                    blue_counts[k5_idx] += 1
                    if blue_counts[k5_idx] == 10:
                        delta_v += 1
                        
            new_v = total_violations + delta_v
            
            accepted = False
            if delta_v <= 0:
                accepted = True
            else:
                if T > 0.005:
                    if random.random() < exp(-delta_v / T):
                        accepted = True
                        
            if accepted:
                colors[e_idx] = new_color
                total_violations = new_v
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_result = colors.copy()
                    
                if total_violations == 0:
                    return colors
            else:
                # Revert
                for k4_idx in edge_to_k4s[e_idx]:
                    if old_color == 0:
                        red_counts[k4_idx] += 1
                    else:
                        red_counts[k4_idx] -= 1
                for k5_idx in edge_to_k5s[e_idx]:
                    if old_color == 1:
                        blue_counts[k5_idx] += 1
                    else:
                        blue_counts[k5_idx] -= 1
                        
            T *= cooling
        
        if best_violations == 0:
            return best_result
    
    return None
