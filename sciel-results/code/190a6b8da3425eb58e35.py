import random
import time
import math

def _energy(sequence, chain):
    """Compute HP contact energy."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x+dx, y+dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _constructive_grow(sequence, n, seed=None):
    """Grow chain residue by residue with energy-aware scoring."""
    if seed is not None:
        random.seed(seed)
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        last_pos = chain[-1]
        prev_pos = chain[-2] if i > 1 else None
        
        candidates = []
        for dx, dy in directions:
            if i > 1:
                if (dx, dy) == (-prev_pos[0] + last_pos[0], -prev_pos[1] + last_pos[1]):
                    continue
            
            new_pos = (last_pos[0] + dx, last_pos[1] + dy)
            if new_pos in occupied:
                continue
            
            next_moves = 0
            for ndx, ndy in directions:
                if (ndx, ndy) == (-dx, -dy):
                    continue
                next_pos = (new_pos[0] + ndx, new_pos[1] + ndy)
                if next_pos not in occupied:
                    next_moves += 1
            
            if next_moves == 0:
                continue
            
            score = 0.0
            if sequence[i] == 'H':
                for dx2, dy2 in directions:
                    nb = (new_pos[0] + dx2, new_pos[1] + dy2)
                    j = occupied.get(nb)
                    if j is not None and j < i and sequence[j] == 'H':
                        score += 2.0
            
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((new_pos[0] - cx)**2 + (new_pos[1] - cy)**2)
            score -= 0.1 * dist
            score += 0.05 * next_moves
            
            candidates.append((score, new_pos))
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            chosen = candidates[0][1]
        else:
            max_score = max(s for s, _ in candidates)
            total = sum(math.exp((s - max_score) / 0.5) for s, _ in candidates)
            r = random.random() * total
            cum = 0
            chosen = candidates[-1][1]
            for s, pos in candidates:
                cum += math.exp((s - max_score) / 0.5)
                if r <= cum:
                    chosen = pos
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain

def _pivot_greedy_polish(sequence, chain, max_rounds=100):
    """Greedily apply pivot moves that improve energy until no more improvements."""
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for rnd in range(max_rounds):
        improved = False
        for k in range(1, n - 1):
            pivot_x, pivot_y = current_chain[k]
            head_positions = set(current_chain[:k+1])
            
            for rot in [1, 2, 3]:
                tail_points = []
                valid = True
                for i in range(k+1, n):
                    px, py = current_chain[i]
                    dx = px - pivot_x
                    dy = py - pivot_y
                    
                    if rot == 1:
                        nx, ny = -dy, dx
                    elif rot == 2:
                        nx, ny = -dx, -dy
                    else:
                        nx, ny = dy, -dx
                    
                    new_pos = (pivot_x + nx, pivot_y + ny)
                    if new_pos in head_positions:
                        valid = False
                        break
                    tail_points.append(new_pos)
                
                if not valid:
                    continue
                
                new_chain = current_chain[:k+1] + tail_points
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
        
        if not improved:
            break
    
    return current_chain

def _generate_snake(n):
    if n == 0:
        return []
    side = int(math.ceil(math.sqrt(n)))
    chain = []
    for row in range(side):
        if row % 2 == 0:
            for col in range(side):
                if len(chain) == n:
                    return chain
                if row == 0 and col == 0:
                    continue
                chain.append((col, row))
        else:
            for col in range(side-1, -1, -1):
                if len(chain) == n:
                    return chain
                chain.append((col, row))
    return chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    deadline = start_time + 190.0
    
    best_chain = _generate_snake(n)
    best_energy = _energy(sequence, best_chain)
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        seed = restart * 12345 + 7
        chain = _constructive_grow(sequence, n, seed=seed)
        
        if chain is None or len(chain) != n:
            chain = _generate_snake(n)
        
        polished = _pivot_greedy_polish(sequence, chain, max_rounds=50)
        energy = _energy(sequence, polished)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = polished
    
    return [list(p) for p in best_chain]
