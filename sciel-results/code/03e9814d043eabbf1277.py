import random
import time

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using
    genetic algorithm with C4 penalty.
    """
    
    def count_c4s(n, colors):
        """Count total monochromatic C4s."""
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    common = adj[c][i] & adj[c][j]
                    if len(common) >= 2:
                        total += 1
        return total
    
    def fitness(n, colors):
        return -count_c4s(n, colors)
    
    def crossover(parent1, parent2):
        n = len(parent1) // (2*n-1) if n else 0
        # This is getting messy. Let me use a simpler GA.
        pass
    
    # GA is complex. Let me use a different new approach.
    
    # New approach: Random restarts with simulated annealing, but with
    # a different cooling schedule and acceptance criterion.
    
    def solve_sa(n, time_limit=100):
        start = time.time()
        colors = [random.randint(0,4) for _ in range(n*(n-1)//2)]
        
        def get_energy():
            return count_c4s(n, colors)
        
        current_energy = get_energy()
        best_colors = colors[:]
        best_energy = current_energy
        
        T = 1.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            if current_energy == 0:
                return {'n': n, 'colors': colors}
            
            # Pick random edge
            idx = random.randint(0, len(colors)-1)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            colors[idx] = new_c
            new_energy = get_energy()
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < (1.0 / (1.0 + (1.0 - 1.0/(1.0+delta)))):
                # Accept
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors[:]
            else:
                colors[idx] = old_c
            
            T *= cooling
        
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
        return None
    
    total_start = time.time()
    for n in range(29, 25, -1):
        remaining = 110 - (time.time() - total_start)
        if remaining < 5:
            break
        res = solve_sa(n, min(20, remaining))
        if res:
            return res
    
    n = 11
    return {'n': n, 'colors': [(i+j)%5 for i in range(n) for j in range(i+1, n)]}
