"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp, sqrt

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2


def _build_contact_graph(sequence):
    """
    Build a conflict graph where nodes are potential H-H contacts.
    Node (i, j) represents a contact between residues i and j (i < j, seq[i]==seq[j]=='H', |i-j|>1).
    Two nodes conflict if they cannot both exist in a valid self-avoiding walk simultaneously.
    For simplicity, we use a relaxed conflict model:
    - Nodes (i,j) and (i,k) conflict if j and k are on the same side of i? No, that's geometry.
    - A simpler, stronger constraint: If we select a set of contacts, we must be able to realize them.
    - This is hard to pre-compute without geometry. 
    - Instead, we will use the graph to guide the search order in a Backtracking approach.
    - We will identify 'high-value' contacts (those involving residues with many potential partners) and prioritize them.
    """
    n = len(sequence)
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Potential contacts: pairs (i, j) with i < j, seq[i]==seq[j]=='H', |i-j|>1
    # We can pre-compute which residues have many H-neighbors in sequence index space
    # to prioritize them.
    
    # For the MIS framing, let's define the 'value' of a contact as 1.
    # The goal is to maximize the number of satisfied contacts.
    
    # We will use a heuristic to order residues for placement:
    # Residues that are 'central' in the H-H interaction graph should be placed first.
    # Degree in potential contact graph: for residue i, count j such that |i-j|>1 and seq[j]=='H'.
    
    degrees = [0] * n
    for i in h_positions:
        for j in h_positions:
            if i != j and abs(i - j) > 1:
                degrees[i] += 1
                
    # Sort residues by degree descending. High degree H's are 'hubs'.
    # We want to place hubs early so we can maximize their contacts.
    
    return degrees


def _backtrack_fold(sequence, dimension, budget_seconds, rng):
    """
    Backtracking search with heuristic ordering.
    We place residues one by one.
    Heuristic:
    1. Start with residue 0 at (0,0).
    2. At each step, we must place the next residue in the chain? 
       No, standard backtracking on the chain sequence is O(4^n).
       Instead, we use a 'growth' backtracking but with strong pruning.
       
       Pruning:
       - Dead-end detection: if the current tip has 0 free neighbors, fail.
       - Contact maximization: prioritize moves that make new H-H contacts.
       - Lookahead: check if the next few residues can be placed.
       
       To make this 'MIS-like', we can view the problem as:
       Select a set of positions for all residues such that:
       1. They form a path.
       2. They are self-avoiding.
       3. The number of H-H adjacencies is maximized.
       
       This is effectively a Maximum Independent Set on a graph of 'conflicting placements'?
       Let's stick to a robust Backtracking with Strong Heuristics.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0) for i in range(n)]
    
    degrees = _build_contact_graph(sequence)
    
    # We will grow the chain.
    # To make it smarter, we can try different 'root' residues?
    # No, the chain order is fixed. Residue 0 must be first.
    
    # However, we can randomize the initial direction and use randomized restarts.
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is better than -1? No, -10 is better than 0.
    # We want to MINIMIZE energy (make it more negative).
    # So best_energy should start at +infinity or 0 (if we assume 0 contacts is baseline).
    # Let's use best_energy = 0 and update if new_energy < best_energy.
    
    start_time = time.time()
    
    # Pre-compute neighbors for fast access
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    def backtrack(chain, occupied, current_energy, depth):
        nonlocal best_chain, best_energy
        
        # Time check
        if time.time() - start_time > budget_seconds:
            return
        
        if depth == n:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return
        
        x, y = chain[-1]
        curr_res = sequence[depth]
        
        # Generate candidates
        candidates = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) in occupied:
                continue
            
            # Calculate gain
            gain = 0
            if curr_res == 'H':
                for ndx, ndy in dirs:
                    nnx, nny = nx + ndx, ny + ndy
                    j = occupied.get((nnx, nny))
                    if j is not None and abs(j - depth) > 1 and sequence[j] == 'H':
                        gain += 1
            
            # Free space
            free = 0
            for ndx, ndy in dirs:
                nnx, nny = nx + ndx, ny + ndy
                if (nnx, nny) not in occupied:
                    free += 1
            
            # Heuristic score
            # Prioritize high gain, then high free space, then avoid traps
            score = gain * 100 + free
            # Trap penalty: if free == 1 and we are not at the end, it's risky
            if free == 1 and depth < n - 2:
                score -= 50
            
            candidates.append((score, (nx, ny), gain))
        
        if not candidates:
            return
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Randomize slightly to escape local minima? 
        # For backtracking, we usually go depth-first.
        # To diversify, we can shuffle the top candidates if they are close in score.
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 10:
            top_k = min(3, len(candidates))
            rng.shuffle(candidates[:top_k])
            # But we still want to try the best ones first.
            # Let's just pick the first one from the shuffled top, or randomize the order of traversal.
            # Actually, for a single backtracking run, we should just pick the best.
            # Randomization should come from RESTARTS with different initial conditions or biases.
            # Here, let's stick to greedy choice for speed, but the 'score' itself is a heuristic.
            # To add randomness, we can use a temperature-like selection.
            # But backtracking is exact (within pruning). 
            # Let's just pick the best.
            pass
            
        # Try the best candidate
        # In a full backtracking, we would try ALL candidates.
        # But 4^n is too big.
        # We need to prune aggressively.
        # This 'backtrack' is actually a 'greedy with backtracking on dead-ends'.
        # If we hit a dead end, we backtrack.
        
        # Let's implement a proper DFS with pruning.
        # But we only have ~100s for 10 sequences. 10s per sequence.
        # 4^100 is impossible.
        # We must use a heuristic search that doesn't explore the whole tree.
        
        # Strategy: 
        # 1. Greedy path.
        # 2. If dead end, backtrack to the last 'branching point' where we had >1 options.
        # 3. Try the next best option at that point.
        
        # This is essentially a depth-first search with a limit on backtracking depth or total nodes visited.
        
        # For now, let's just do a greedy walk with backtracking on immediate failure.
        # This is similar to the current program but with the 'degrees' info potentially used?
        # The 'degrees' info isn't used in this simple growth.
        
        # Let's refine: 
        # We will use a 'beam search' or 'limited backtracking'.
        
        # For this implementation, let's stick to a robust greedy with random restarts and local search.
        # The 'MIS' framing is too complex to implement efficiently in this time limit without a specific library.
        # I will implement a stronger local search with 'pivot moves' (rotating a sub-chain).
        
        return None # Signal to use local search instead
        
    # The above backtrack function is a stub. 
    # Let's implement the actual strategy: 
    # 1. Generate initial conformations using a smarter greedy (using degrees to bias H placement).
    # 2. Improve with Pivot Moves (local search).
    
    # Pivot Move: 
    # Select a segment of the chain [i, j].
    # Rotate the segment by 90, 180, 270 degrees around residue i-1 or j+1?
    # Or simply: select i, j. Try to move the subchain (i..j) to a new position 
    # such that it remains connected to i-1 and j+1 and is self-avoiding.
    # This is expensive.
    
    # Let's stick to Single Residue Moves (as in current) but with a better acceptance criterion 
    # and more restarts.
    
    # Actually, the current program's weakness is likely the 'greedy' growth getting stuck.
    # Let's improve the growth to be 'lookahead' based.
    
    def _grow_lookahead(sequence, rng):
        n = len(sequence)
        if n <= 2:
            return [(i, 0) for i in range(n)]
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        for i in range(1, n):
            candidates = []
            x, y = chain[-1]
            
            for dx, dy in dirs:
                new_pt = (x + dx, y + dy)
                if new_pt in occupied:
                    continue
                
                # Lookahead: can we place the NEXT residue?
                next_res = sequence[i+1] if i+1 < n else 'P'
                next_ok = False
                if i+1 < n:
                    for ndx, ndy in dirs:
                        nnp = (new_pt[0]+ndx, new_pt[1]+ndy)
                        if nnp not in occupied and nnp != chain[-2]: # nnp != chain[-2] is not strictly necessary, just not occupied
                            # Also check if nnp is a dead end for the one after?
                            # Just check if it's free.
                            next_ok = True
                            break
                
                if not next_ok and i+1 < n:
                    # This move leads to a dead end for the next residue.
                    # Penalize heavily.
                    score = -1000
                else:
                    # Calculate gain
                    gain = 0
                    if sequence[i] == 'H':
                        for ndx, ndy in dirs:
                            nnx, nny = new_pt[0]+ndx, new_pt[1]+ndy
                            j = occupied.get((nnx, nny))
                            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                                gain += 1
                    
                    free = 0
                    for ndx, ndy in dirs:
                        nnx, nny = new_pt[0]+ndx, new_pt[1]+ndy
                        if (nnx, nny) not in occupied:
                            free += 1
                            
                    score = gain * 10 + free
                    
                candidates.append((score, new_pt))
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick best, or randomize if close
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 5:
                choice = rng.choice(candidates[:min(3, len(candidates))])
            else:
                choice = candidates[0]
                
            chain.append(choice[1])
            occupied[choice[1]] = i
            
        return chain

    # Phase 1: Generate candidates
    candidates = []
    num_restarts = 10
    for _ in range(num_restarts):
        if _remaining_time() < 15.0:
            break
        c = _grow_lookahead(sequence, rng)
        if c:
            candidates.append(c)
            
    if not candidates:
        return [(i, 0) for i in range(n)]
        
    # Phase 2: Local Search (Single Residue Moves)
    best_chain = None
    best_energy = 0
    
    # Sort candidates by energy
    scored = [(_energy(sequence, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    
    # Take top 3
    top_candidates = scored[:3]
    
    time_per_candidate = budget_seconds / len(top_candidates)
    
    for e, c in top_candidates:
        if _remaining_time() < 2.0:
            break
            
        current_chain = c
        current_energy = e
        occupied = {p: i for i, p in enumerate(current_chain)}
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        start_t = time.time()
        
        while time.time() - start_t < time_per_candidate:
            idx = rng.randint(1, n-2)
            dx, dy = rng.choice(dirs)
            
            old_pt = current_chain[idx]
            new_pt = (old_pt[0]+dx, old_pt[1]+dy)
            
            if new_pt in occupied:
                # Check if it's a neighbor
                occupant = occupied[new_pt]
                if abs(occupant - idx) > 1:
                    continue # Collision
                else:
                    continue # Can't move into bond
            
            # Check connectivity
            prev_pt = current_chain[idx-1]
            next_pt = current_chain[idx+1]
            if abs(new_pt[0]-prev_pt[0]) + abs(new_pt[1]-prev_pt[1]) != 1:
                continue
            if abs(new_pt[0]-next_pt[0]) + abs(new_pt[1]-next_pt[1]) != 1:
                continue
            
            # Calculate energy change
            delta = 0
            if sequence[idx] == 'H':
                # Remove old contacts
                for ndx, ndy in dirs:
                    nnx, nny = old_pt[0]+ndx, old_pt[1]+ndy
                    j = occupied.get((nnx, nny))
                    if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                        delta -= 1 # Lose contact -> energy increases (delta E > 0)
                        
                # Add new contacts
                for ndx, ndy in dirs:
                    nnx, nny = new_pt[0]+ndx, new_pt[1]+ndy
                    j = occupied.get((nnx, nny))
                    if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                        delta += 1 # Gain contact -> energy decreases (delta E < 0)
                        
            # Note: The sign convention in _energy is -contacts.
            # So if we gain 1 contact, energy becomes more negative by 1.
            # delta_energy = - (new_contacts - old_contacts)
            # In my loop above:
            # If I lose a contact, delta (contacts) = -1. delta_energy = -(-1) = +1.
            # If I gain a contact, delta (contacts) = +1. delta_energy = -(+1) = -1.
            # So my `delta` variable above is actually `delta_contacts`.
            # delta_energy = -delta.
            
            delta_E = -delta
            
            if delta_E <= 0 or rng.random() < exp(-delta_E / T):
                # Accept
                current_chain[idx] = new_pt
                del occupied[old_pt]
                occupied[new_pt] = idx
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
    if best_chain is None:
        best_chain = top_candidates[0][1]
        
    return best_chain


def fold(sequence, dimension):
    """Best conformation found via Lookahead Growth + Simulated Annealing."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time
    # We have ~120s total for 10 sequences.
    # So ~12s per sequence.
    # Let's give 10s to this function.
    
    chain = _backtrack_fold(sequence, dimension, 10.0, rng)
    
    if chain is None:
        # Fallback
        return [[i, 0] for i in range(n)]
        
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
