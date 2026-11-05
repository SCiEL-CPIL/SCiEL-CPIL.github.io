"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import time

# Precompute grid points and isosceles triples
_points = list(itertools.product(range(N), repeat=2))
_index_map = {p: i for i, p in enumerate(_points)}
_num_points = len(_points)

# Precompute all isosceles triples
# To save memory/time, we can store them as tuples of indices.
# Number of triples is ~2.7M. Storing them might be heavy but acceptable.
# Alternatively, we can compute on the fly if needed, but SA needs speed.
# Let's precompute a list of sets of 3 indices.
_isosceles_triples = []
# Optimization: Only store triples.
# We can also precompute for each point p, the list of pairs (a,b) such that (p,a,b) is isosceles.
# This helps in checking maximality quickly.
# adj[p] = list of (a, b) where a < b and (p,a,b) is isosceles.

# Let's build adjacency structure for faster checks
# For each point p, we want to know which other points q are "compatible" in the sense that they don't form isosceles with a third point?
# No, the constraint is on triples.

# Let's try a different representation for speed:
# For each pair (a, b), find all c such that (a,b,c) is isosceles.
# Let's just precompute the list of triples. It's the most straightforward.

def _precompute_triples():
    triples = []
    pts = _points
    for i in range(_num_points):
        for j in range(i + 1, _num_points):
            for k in range(j + 1, _num_points):
                a, b, c = pts[i], pts[j], pts[k]
                d_ab = (a[0]-b[0])**2 + (a[1]-b[1])**2
                d_bc = (b[0]-c[0])**2 + (b[1]-c[1])**2
                d_ca = (c[0]-a[0])**2 + (c[1]-a[1])**2
                if d_ab == d_bc or d_bc == d_ca or d_ca == d_ab:
                    triples.append((i, j, k))
    return triples

_triples = _precompute_triples()

def _eval_state(state_set):
    """
    Returns (violations, unblocked_count, size)
    violations: number of isosceles triples fully in state_set
    unblocked_count: number of points NOT in state_set that can be added (i.e., no isosceles triple with 2 in state_set)
    """
    size = len(state_set)
    
    # Check violations
    violations = 0
    for t in _triples:
        if t[0] in state_set and t[1] in state_set and t[2] in state_set:
            violations += 1
            if violations > 0: # Early exit? No, we need count for SA? 
                # Actually, for SA, we just need to know if it's 0 or not, 
                # but penalty scaling helps. Let's just count up to a limit.
                pass
                
    # Check maximality
    # A point p is addable if there is NO triple (p, a, b) with a,b in state_set.
    # Equivalent: p is blocked if there EXISTS at least one triple (p, a, b) with a,b in state_set.
    # We need count of p NOT in state_set such that NO such triple exists.
    
    # Optimization: Precompute for each p, the list of pairs (a,b) that form isosceles with p.
    # Let's build this map once.
    
    unblocked = 0
    for p_idx in range(_num_points):
        if p_idx in state_set:
            continue
        # Check if p is blocked
        # p is blocked if any (a,b) in _adj[p_idx] are both in state_set
        is_blocked = False
        for a, b in _adj[p_idx]:
            if a in state_set and b in state_set:
                is_blocked = True
                break
        if not is_blocked:
            unblocked += 1
            
    return violations, unblocked, size

# Build adjacency map: _adj[p] = list of (a, b) such that (p, a, b) is isosceles
_adj = [[] for _ in range(_num_points)]
for i, j, k in _triples:
    # i, j, k form isosceles
    _adj[i].append((j, k))
    _adj[j].append((i, k))
    _adj[k].append((i, j))

def _sa_construct():
    start_time = time.time()
    max_time = 290 # Leave 10s buffer
    
    # Initial state: Random small set? Or empty?
    # Empty set has 0 violations, but many unblocked.
    # Let's start with a random set of size ~10.
    initial_size = 10
    current_state = set(random.sample(range(_num_points), initial_size))
    
    current_viol, current_unblocked, current_size = _eval_state(current_state)
    current_cost = current_size + 1000 * current_viol + 100 * current_unblocked
    
    best_state = set(current_state)
    best_cost = current_cost
    
    T = 1000.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() - start_time < max_time:
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.3:
            # Add a point
            candidates = [i for i in range(_num_points) if i not in current_state]
            if not candidates: continue
            p = rng.choice(candidates)
            new_state = set(current_state)
            new_state.add(p)
        elif move_type < 0.6:
            # Remove a point
            if not current_state: continue
            p = rng.choice(list(current_state))
            new_state = set(current_state)
            new_state.remove(p)
        else:
            # Swap
            if not current_state: continue
            p_in = rng.choice(list(current_state))
            candidates = [i for i in range(_num_points) if i not in current_state]
            if not candidates: continue
            p_out = rng.choice(candidates)
            new_state = set(current_state)
            new_state.remove(p_in)
            new_state.add(p_out)
            
        new_viol, new_unblocked, new_size = _eval_state(new_state)
        new_cost = new_size + 1000 * new_viol + 100 * new_unblocked
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < pow(2.718, -delta / T):
            current_state = new_state
            current_viol, current_unblocked, current_size = new_viol, new_unblocked, new_size
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = set(current_state)
                
        T *= cooling
        
    return list(best_state)

def construct():
    # Run SA for a bit
    # Convert indices back to points
    indices = _sa_construct()
    return [_points[i] for i in indices]
# EVOLVE-BLOCK-END
