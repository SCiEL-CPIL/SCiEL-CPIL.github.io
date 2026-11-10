"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk (random restart)."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None
        
        next_site = rng.choice(options)
        chain.append(next_site)
        occupied[next_site] = i
        
    return chain


def _min_conflicts_search(sequence, chain, rng, max_iters=5000):
    """
    Min-Conflicts style local search.
    We treat the energy minimization as a constraint satisfaction problem.
    Moves are:
    1. Kink flips (changing a 90-degree turn to the other 90-degree turn).
    2. Pivot moves (rotating a sub-chain around a pivot point).
    We accept moves that reduce the "conflict" (energy) or with some probability
    accept neutral/worse moves to escape local minima (simulated annealing hybrid).
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temp = 1.0
    cool_rate = 0.999
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        if temp < 0.01:
            break
            
        # Choose a move type
        # Kink flips are cheap and frequent. Pivots are expensive but powerful.
        if n > 10 and rng.random() < 0.2:
            # Pivot move
            # Pick a pivot index i (1 to n-3)
            i = rng.randint(1, n - 3)
            # The sub-chain is i+1 to j. Let's pick a short sub-chain for speed
            # or a random length.
            max_len = min(5, n - i - 1)
            if max_len < 1:
                continue
            length = rng.randint(1, max_len)
            j = i + length
            
            # The sub-chain is chain[i+1 ... j]
            # We rotate it 90 degrees clockwise or counter-clockwise around chain[i]
            # But we must ensure the new positions are not occupied by the rest of the chain.
            
            sub_chain = current_chain[i+1:j+1]
            pivot = current_chain[i]
            
            # Generate new positions for the sub-chain
            # Rotation of a vector (dx, dy) 90 deg CW: (dy, -dx)
            # Rotation 90 deg CCW: (-dy, dx)
            
            def rotate_cw(pt, piv):
                dx = pt[0] - piv[0]
                dy = pt[1] - piv[1]
                return (piv[0] + dy, piv[1] - dx)
                
            def rotate_ccw(pt, piv):
                dx = pt[0] - piv[0]
                dy = pt[1] - piv[1]
                return (piv[0] - dy, piv[1] + dx)
            
            # We need to check if the new positions are free.
            # The "rest" of the chain is current_chain[:i+1] + current_chain[j+1:]
            # Note: current_chain[i] is the pivot, which stays.
            # The positions occupied by the rest are:
            occupied_rest = set(current_chain[:i+1] + current_chain[j+1:])
            # Remove the positions currently occupied by the sub-chain from the "forbidden" set?
            # No, the sub-chain moves away, so its old positions become free.
            # The new positions must not be in occupied_rest.
            
            new_sub_cw = [rotate_cw(p, pivot) for p in sub_chain]
            new_sub_ccw = [rotate_ccw(p, pivot) for p in sub_chain]
            
            valid_cw = all(p not in occupied_rest for p in new_sub_cw)
            valid_ccw = all(p not in occupied_rest for p in new_sub_ccw)
            
            if not valid_cw and not valid_ccw:
                no_improve += 1
                temp *= cool_rate
                continue
                
            if valid_cw and valid_ccw:
                choice = rng.choice([0, 1])
            elif valid_cw:
                choice = 0
            else:
                choice = 1
                
            if choice == 0:
                new_sub = new_sub_cw
            else:
                new_sub = new_sub_ccw
                
            new_chain = current_chain[:i+1] + new_sub + current_chain[j+1:]
            
        else:
            # Kink flip
            # Find a kink: i-1, i, i+1 where dist(i-1, i+1) == 2 (they are neighbours)
            # Actually, on a lattice, if A-B-C is a path, and A and C are neighbours,
            # then B is the "corner". Flipping means moving B to the 4th point that
            # completes the square.
            
            i = rng.randint(1, n - 2)
            p_prev = current_chain[i-1]
            p_curr = current_chain[i]
            p_next = current_chain[i+1]
            
            # Check if it's a kink
            if abs(p_prev[0] - p_next[0]) + abs(p_prev[1] - p_next[1]) != 2:
                # Not a kink (it's a straight line or a U-turn which is impossible in SAW)
                # Wait, in a SAW, A-B-C cannot be a U-turn (A=C).
                # If dist(A,C) == 1, it's a kink.
                # If dist(A,C) == 2, it's a straight line.
                # The 4th point is only valid if it's a kink.
                # If it's a straight line, we can't "flip" to a 4th point that keeps connectivity?
                # Actually, if A-B-C is straight, the "flip" would be moving B to a side,
                # which breaks the A-C connection? No, A and C are not connected.
                # But moving B to a side makes A-B-C a kink.
                # Let's only do flips on existing kinks for simplicity and safety.
                no_improve += 1
                temp *= cool_rate
                continue
                
            # It is a kink. Find the 4th point.
            # A = (ax, ay), B = (bx, by), C = (cx, cy)
            # The 4th point D is such that A-B-C-D is a square.
            # D = A + C - B
            d_x = p_prev[0] + p_next[0] - p_curr[0]
            d_y = p_prev[1] + p_next[1] - p_curr[1]
            new_pos = (d_x, d_y)
            
            # Check if new_pos is occupied by any other part of the chain
            # It must not be in current_chain except at index i (which we are moving)
            # And it must not be p_prev or p_next (it won't be, as it's a square)
            
            is_occupied = False
            for k, p in enumerate(current_chain):
                if k == i:
                    continue
                if p == new_pos:
                    is_occupied = True
                    break
                    
            if is_occupied:
                no_improve += 1
                temp *= cool_rate
                continue
                
            new_chain = list(current_chain)
            new_chain[i] = new_pos
            
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta / temp)
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= cool_rate
        
        if no_improve > 500:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using Min-Conflicts style local search.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    n = len(sequence)
    
    # Adaptive time allocation
    # Min-Conflicts is very fast per step, so we can do many restarts.
    
    # Phase 1: Generate a good initial solution using a few random walks or greedy
    # We'll use a simple greedy to get a valid start, then optimize.
    # If greedy fails, use random walk.
    
    # Let's do multiple independent runs of:
    # 1. Generate initial chain (Greedy or Random)
    # 2. Min-Conflicts optimization
    
    runs = 0
    max_runs = 1000
    
    # Time budget for the whole process
    start_time = time.time()
    total_time_budget = _DEADLINE - start_time
    
    # We will spend most of the time in optimization.
    # Let's allocate 80% to optimization, 20% to initial generation/restarts.
    # But since initial generation is fast, we just loop until time is up.
    
    for run in range(max_runs):
        if time.time() > _DEADLINE - 10:
            break
        
        # 1. Generate initial chain
        # Try greedy first, fallback to random
        # We don't have the old greedy function here, so let's implement a simple one
        # or just use random walk which is guaranteed to be valid if it completes.
        # Random walks often get stuck for long sequences.
        # Let's use a simple "biased random walk" or just random.
        
        chain = _random_walk(sequence, rng)
        if chain is None:
            # Random walk failed (stuck). Try a different start or just skip.
            # For long sequences, random walk fails often.
            # We should probably have a better initializer.
            # Let's implement a simple greedy initializer here.
            chain = _simple_greedy(sequence, rng)
            if chain is None:
                continue
                
        # 2. Optimize
        # Allocate time to this run
        # If we have 100s left, and we do 100 runs, each gets 1s.
        # Min-Conflicts is fast.
        
        # Let's just run with a fixed number of iterations or time limit.
        # Since we don't know how many runs we'll do, let's just run with a fixed
        # max_iters and rely on the outer loop.
        
        opt_chain, opt_energy = _min_conflicts_search(sequence, chain, rng, max_iters=2000)
        
        if best_chain is None or opt_energy < best_energy:
            best_chain = opt_chain
            best_energy = opt_energy
            
        runs += 1
        
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]


def _simple_greedy(sequence, rng, temperature=0.5):
    """Simple greedy growth with randomization."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
        
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
            
        if not options:
            return None
            
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain
# EVOLVE-BLOCK-END
