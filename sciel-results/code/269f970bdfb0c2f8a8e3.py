import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all K4s
    # Edge index for (u, v) with u < v:
    # Number of edges before u: sum_{k=0}^{u-1} (n-1-k) = u*(n-1) - u*(u-1)//2
    # Offset within row u: v - u - 1
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    num_k4 = len(k4_edges)

    def calculate_penalty(coloring, max_colors):
        """
        Returns the total penalty.
        Penalty for a K4 is max(0, 5 - num_distinct_colors).
        """
        total_penalty = 0
        for edges in k4_edges:
            c0 = coloring[edges[0]]
            c1 = coloring[edges[1]]
            c2 = coloring[edges[2]]
            c3 = coloring[edges[3]]
            c4 = coloring[edges[4]]
            c5 = coloring[edges[5]]
            
            # Count distinct colors
            # Since max_colors is small (<=15), we can use a small set or bitmask
            # Bitmask is faster
            mask = 0
            mask |= 1 << c0
            mask |= 1 << c1
            mask |= 1 << c2
            mask |= 1 << c3
            mask |= 1 << c4
            mask |= 1 << c5
            
            num_distinct = bin(mask).count('1')
            if num_distinct < 5:
                total_penalty += (5 - num_distinct)
                
        return total_penalty

    def is_valid(coloring):
        return calculate_penalty(coloring, 0) == 0

    # Initial random coloring
    current_max_colors = 15
    coloring = [random.randint(0, current_max_colors - 1) for _ in range(num_edges)]
    
    best_penalty = calculate_penalty(coloring, current_max_colors)
    best_coloring = coloring.copy()
    
    # Simulated Annealing
    T = 100.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    while T > min_T and time.time() < deadline - 5.0:
        # Pick a random edge to change
        e_idx = random.randint(0, num_edges - 1)
        old_color = coloring[e_idx]
        new_color = random.randint(0, current_max_colors - 1)
        
        if old_color == new_color:
            continue
            
        coloring[e_idx] = new_color
        new_penalty = calculate_penalty(coloring, current_max_colors)
        
        delta = new_penalty - best_penalty
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            best_penalty = new_penalty
            best_coloring = coloring.copy()
        else:
            coloring[e_idx] = old_color
            
        T *= cooling_rate
        
        # If we found a valid solution, try to reduce colors immediately
        if best_penalty == 0:
            # Try to reduce colors
            reduced = False
            # Try to eliminate the highest color
            color_to_eliminate = current_max_colors - 1
            edges_to_change = [i for i, c in enumerate(best_coloring) if c == color_to_eliminate]
            
            if edges_to_change:
                # Try to recolor these edges to lower colors
                success = False
                for _ in range(50):
                    trial = best_coloring.copy()
                    for e in edges_to_change:
                        trial[e] = random.randint(0, current_max_colors - 2)
                    if is_valid(trial):
                        success = True
                        best_coloring = trial
                        current_max_colors -= 1
                        reduced = True
                        break
                
                if success:
                    # Reset SA to continue optimizing or reducing further
                    coloring = best_coloring.copy()
                    best_penalty = 0
                    T = 10.0 # Lower temperature for fine-tuning
                    continue
            
            # If we can't reduce, keep searching with current max to see if we find a better structure
            # Or just break if we are at 13?
            # The task says minimize. 13 is the lower bound.
            if current_max_colors <= 13:
                break

    # If we found a valid coloring, return it
    if best_penalty == 0:
        return best_coloring
    
    # If we didn't find a valid one, we need to return something.
    # The seed returns a specific pattern. We should return a valid one if possible.
    # A known valid construction for K16 with 15 colors?
    # Let's try a structured approach if SA fails.
    # Actually, random SA with 15 colors should find a valid one quickly.
    # If not, we return the best effort, but it must be valid.
    # Let's force a valid solution using a greedy repair if SA fails.
    
    # Fallback: Greedy construction
    # This is rarely needed but ensures validity.
    # We can use the result from a known paper or just a robust random restart.
    
    # Let's do a few quick random restarts if we failed
    for restart in range(10):
        coloring = [random.randint(0, 14) for _ in range(num_edges)]
        T = 50.0
        for _ in range(1000):
            e_idx = random.randint(0, num_edges - 1)
            old_color = coloring[e_idx]
            new_color = random.randint(0, 14)
            if old_color == new_color: continue
            coloring[e_idx] = new_color
            new_penalty = calculate_penalty(coloring, 15)
            delta = new_penalty - best_penalty
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                best_penalty = new_penalty
                best_coloring = coloring.copy()
            else:
                coloring[e_idx] = old_color
            T *= 0.999
        if best_penalty == 0:
            return best_coloring
            
    # If still invalid, return a hardcoded valid solution if we had one, 
    # or the seed's fallback (which is 17 colors, valid? (i+j)%17).
    # The seed fallback is: [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Let's verify if (i+j)%17 is valid.
    # It uses 17 colors.
    # We must return a valid coloring.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
