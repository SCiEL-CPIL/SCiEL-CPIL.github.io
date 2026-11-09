"""
Strategy: Path Reversal Simulated Annealing for HP Lattice Folding.
Different from the seed's local-move SA, this uses segment reversals (flips).
Reversals maintain self-avoidance and connectivity automatically if the segment
is valid, but allow for significant rearrangement of the chain topology.
"""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


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


def _grow_greedy(sequence, dimension, rng, bias=0.5):
    """
    Greedy growth to create a valid initial self-avoiding walk.
    Prioritizes moves that create H-H contacts while avoiding dead ends.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0) for i in range(n)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for i in range(1, n):
        candidates = []
        x, y = chain[-1]
        
        for dx, dy in dirs:
            new_pt = (x + dx, y + dy)
            if new_pt in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                               (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                    q = (nx, ny)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = 0
            for nx, ny in [(new_pt[0]+1, new_pt[1]), (new_pt[0]-1, new_pt[1]), 
                           (new_pt[0], new_pt[1]+1), (new_pt[0], new_pt[1]-1)]:
                if (nx, ny) not in occupied:
                    free += 1
            
            penalty = 0
            if free == 1 and i < n - 2:
                penalty = 2.0
            
            score = (gain * 10.0) + (free * bias) - penalty
            candidates.append((score, new_pt))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 1.0:
            top_k = min(3, len(candidates))
            choice = rng.choice(candidates[:top_k])
        else:
            choice = candidates[0]
            
        new_pt = choice[1]
        occupied[new_pt] = i
        chain.append(new_pt)
        
    return chain


def _calculate_energy_delta(sequence, chain, i, j):
    """
    Calculate the change in energy if we reverse the segment from i to j (inclusive).
    A reversal affects contacts involving residues in [i, j] with residues outside [i, j].
    Specifically, the neighbors of the segment ends (i-1, j+1) and the internal structure.
    
    Actually, for a reversal of [i, j]:
    - Residues i..j keep their relative positions but reversed order? No, positions are fixed on lattice.
    - Wait, a "reversal" in lattice folding usually means:
      Take the path segment from i to j. Reverse the direction of the path.
      This is only possible if the segment forms a "loop" or if we are doing a "flip" in 3D?
      In 2D, a simple reversal of indices does NOT change the set of occupied lattice points.
      However, it DOES change the sequence-to-position mapping.
      
      Let's clarify:
      Chain is a list of points P[0], P[1], ..., P[n-1].
      Sequence is S[0], S[1], ..., S[n-1].
      Energy depends on pairs (a, b) such that dist(P[a], P[b]) == 1 and |a-b| > 1 and S[a]=='H' and S[b]=='H'.
      
      If we reverse the sequence segment S[i..j] to S[j], S[j-1]...S[i], the positions P[i..j] remain the same.
      The contacts between two residues both inside [i, j] remain the same (distances unchanged, H/P status of the *positions* unchanged? No, the H/P status moves with the sequence index).
      
      Let's re-evaluate.
      Standard "Reversal" move in HP folding:
      Often refers to reversing the *direction* of the chain segment, which in 2D is tricky without breaking self-avoidance unless it's a specific topological move.
      
      Alternative interpretation: "Flip" or "Rotate" a sub-chain?
      
      Actually, the most common "global" move that is easy to check for validity in 2D is the **Pivot** (already in seed) or **Reversal of the path direction**.
      
      Let's look at the "Reversal" definition in the context of the inspiration:
      "uses ... _apply_reversal ...".
      
      If we reverse the segment of the *chain* (the geometric path), i.e., we take the points P[i], P[i+1]...P[j] and replace them with P[j], P[j-1]...P[i].
      Does this maintain self-avoidance?
      The set of points is identical. So no new collisions are introduced *within* the segment.
      Collisions with the *rest* of the chain:
      The rest of the chain is P[0..i-1] and P[j+1..n-1].
      The segment endpoints change:
      Old endpoints: P[i] (connected to P[i-1]) and P[j] (connected to P[j+1]).
      New endpoints: P[j] (connected to P[i-1]) and P[i] (connected to P[j+1]).
      
      So, for the move to be valid:
      1. P[j] must be adjacent to P[i-1] (unit step).
      2. P[i] must be adjacent to P[j+1] (unit step).
      3. The new segment P[j]...P[i] must not collide with P[0..i-1] or P[j+1..n-1].
         Since the set of points in the segment is the same, and the segment was valid before, it won't collide with *itself*.
         It only collides with the outside if P[j] (new start) is adjacent to some P[k] (k < i-1) that wasn't adjacent to P[i] (old start)? No, self-avoidance is about occupancy.
         The occupancy set is identical. So self-avoidance is automatically preserved IF the new connections don't force an overlap?
         Wait, if the set of points is the same, the lattice is occupied by the same points.
         The only thing that changes is the *sequence* of points.
         So, is it always self-avoiding? Yes, because the set of occupied lattice sites is identical.
         
         So the ONLY validity checks are:
         1. dist(P[i-1], P[j]) == 1
         2. dist(P[j+1], P[i]) == 1
         
         This is a very cheap validity check! O(1).
         
         Now, energy change.
         Energy = - (H-H contacts).
         A contact is a pair (a, b) with |a-b|>1, dist(P[a], P[b])==1, S[a]=='H', S[b]=='H'.
         
         When we reverse S[i..j], the mapping from index to position changes for indices in [i, j].
         Let pos(k) be the lattice point at index k.
         Old: pos_old(k) = P[k] for k in [i, j].
         New: pos_new(k) = P[i + j - k] for k in [i, j].
         
         Contacts are determined by:
         1. Two H's.
         2. Their positions are adjacent.
         3. Their indices are not adjacent.
         
         Let's analyze which contacts change.
         Contacts involving only indices outside [i, j]: Unchanged.
         Contacts involving one index inside [i, j] and one outside:
         Let a in [i, j] and b outside.
         Old: dist(P[a], P[b]) == 1?
         New: dist(P_new(a), P[b]) == 1? where P_new(a) = P[i+j-a].
         So we need to check if the new position of residue a is adjacent to residue b.
         
         Contacts involving two indices inside [i, j]:
         Let a, b in [i, j].
         Old: dist(P[a], P[b]) == 1 and |a-b|>1.
         New: dist(P_new(a), P_new(b)) == 1 and |a-b|>1.
         P_new(a) = P[i+j-a], P_new(b) = P[i+j-b].
         dist(P[i+j-a], P[i+j-b]) == dist(P[a], P[b])?
         Yes, because distance is symmetric and the set of points is the same, just mapped differently.
         Specifically, the pair of points (P[a], P[b]) is replaced by (P[i+j-a], P[i+j-b]).
         Are these the same pair of points? Not necessarily.
         However, the *set* of pairs of points in the segment is the same.
         Wait. The contact condition is:
         1. Points are adjacent.
         2. Indices are not adjacent.
         3. Both are H.
         
         In the new configuration:
         The point at index a is now P[i+j-a].
         The point at index b is now P[i+j-b].
         The distance is dist(P[i+j-a], P[i+j-b]).
         This is the same as dist(P[a], P[b]) ONLY if the mapping preserves distances? No.
         The mapping is a reflection.
         The set of points occupied by the segment is identical.
         The set of *pairs* of points that are adjacent in the lattice is identical.
         Let E be the set of adjacent pairs of points in the segment.
         In the old chain, a contact exists if there is a pair (u, v) in E such that u=P[a], v=P[b], |a-b|>1, S[a]=='H', S[b]=='H'.
         In the new chain, the point u=P[a] is now at index a' = i+j-a.
         The point v=P[b] is now at index b' = i+j-b.
         The contact exists if u and v are adjacent (true, they are in E), |a'-b'|>1, S[a'] (which is S[i+j-a])=='H', S[b'] (which is S[i+j-b])=='H'.
         
         So, the energy change depends on:
         1. The H/P status of the sequence at the new indices.
         2. The index distance |a'-b'|.
         
         This is complex to calculate incrementally for all pairs.
         However, we can just recalculate the energy for the whole chain.
         For N=100, _energy is O(N^2) or O(N * neighbors).
         _energy is fast enough.
         
         So the strategy:
         1. Generate initial chain.
         2. Anneal loop:
            - Pick i, j randomly (i < j).
            - Check validity: dist(P[i-1], P[j]) == 1 and dist(P[j+1], P[i]) == 1.
            - If valid:
               - Reverse chain[i:j+1].
               - Calculate new energy.
               - Accept/Reject based on SA.
            
         This is much simpler than local moves and allows large jumps.
    """
    n = len(chain)
    if i <= 0 or j >= n - 1:
        return 0  # Can't reverse ends as they have only 1 neighbor
    
    # Check connectivity validity
    # P[i-1] must be adjacent to P[j] (new start of segment)
    # P[j+1] must be adjacent to P[i] (new end of segment)
    
    p_i_minus_1 = chain[i-1]
    p_j = chain[j]
    p_j_plus_1 = chain[j+1]
    p_i = chain[i]
    
    if abs(p_i_minus_1[0] - p_j[0]) + abs(p_i_minus_1[1] - p_j[1]) != 1:
        return 0
    if abs(p_j_plus_1[0] - p_i[0]) + abs(p_j_plus_1[1] - p_i[1]) != 1:
        return 0
        
    # Validity check passed.
    # Now calculate energy delta.
    # For simplicity and robustness, recalculate energy.
    # Optimization: Only recalculate if we are going to accept? No, we need delta for acceptance.
    # We can compute old_energy (known) and new_energy (calc).
    
    # Perform reversal on a copy
    new_chain = chain[:]
    new_chain[i:j+1] = reversed(new_chain[i:j+1])
    
    new_e = _energy(sequence, new_chain)
    return new_e - _energy(sequence, chain) # This calculates old energy twice, bad.
    
    # Let's pass current energy.
    # Revised function signature: _try_reversal(sequence, chain, i, j, current_energy)
    
    # But wait, the function above returns delta. I need to fix the logic inside.
    # I will implement the logic directly in the anneal loop for speed, or a helper that returns (valid, new_chain, new_energy).

def _try_reversal(sequence, chain, i, j, current_energy):
    """
    Attempt to reverse chain[i:j+1].
    Returns (success, new_chain, new_energy).
    """
    n = len(chain)
    if i <= 0 or j >= n - 1:
        return False, None, None
    
    p_i_minus_1 = chain[i-1]
    p_j = chain[j]
    p_j_plus_1 = chain[j+1]
    p_i = chain[i]
    
    # Check if new connections are valid unit steps
    if abs(p_i_minus_1[0] - p_j[0]) + abs(p_i_minus_1[1] - p_j[1]) != 1:
        return False, None, None
    if abs(p_j_plus_1[0] - p_i[0]) + abs(p_j_plus_1[1] - p_i[1]) != 1:
        return False, None, None
        
    # Construct new chain
    new_chain = chain[:]
    new_chain[i:j+1] = reversed(new_chain[i:j+1])
    
    # Calculate new energy
    new_energy = _energy(sequence, new_chain)
    
    return True, new_chain, new_energy


def _anneal_reversal(sequence, initial_chain, budget_seconds, rng):
    """
    Simulated Annealing using Reversal moves.
    """
    n = len(sequence)
    if n <= 2:
        return initial_chain
    
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_t = time.time()
    no_improve = 0
    max_no_improve = 5000
    
    while _remaining_time() > budget_seconds:
        if no_improve > max_no_improve:
            break
            
        # Choose segment to reverse
        # Bias towards shorter segments? Or random?
        # Shorter segments are more likely to be valid (endpoints adjacent).
        # Let's try lengths from 2 to n//2.
        length = rng.randint(2, max(2, n // 3))
        i = rng.randint(1, n - length - 1)
        j = i + length - 1
        
        success, new_chain, new_energy = _try_reversal(sequence, current_chain, i, j, current_energy)
        
        if not success:
            no_improve += 1
            continue
            
        delta_E = new_energy - current_energy
        
        if delta_E <= 0 or rng.random() < exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = current_chain
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_chain


def fold(sequence, dimension):
    """
    Fold sequence using Greedy Growth + Reversal Annealing.
    """
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(54321 + len(sequence))
    
    candidates = []
    
    # Phase 1: Generate diverse initial conformations
    biases = [0.2, 0.5, 1.0, 2.0, 5.0]
    num_attempts = 5
    
    for bias in biases:
        for _ in range(num_attempts):
            if _remaining_time() < 10.0:
                break
            chain = _grow_greedy(sequence, dimension, rng, bias)
            if chain is not None:
                candidates.append(chain)
        
        if _remaining_time() < 10.0:
            break
    
    if not candidates:
        return [[i, 0] for i in range(n)]
    
    # Evaluate and sort
    candidates_with_energy = []
    for c in candidates:
        e = _energy(sequence, c)
        candidates_with_energy.append((e, c))
    
    candidates_with_energy.sort(key=lambda x: x[0])
    
    best_energy = candidates_with_energy[0][0]
    best_chain = candidates_with_energy[0][1]
    
    # Phase 2: Anneal top candidates with Reversal moves
    num_to_anneal = min(3, len(candidates_with_energy))
    
    remaining_time = _remaining_time()
    search_time = max(1.0, remaining_time - 5.0)
    
    if num_to_anneal == 1:
        budgets = [search_time]
    elif num_to_anneal == 2:
        budgets = [search_time * 0.6, search_time * 0.4]
    else:
        budgets = [search_time * 0.5, search_time * 0.3, search_time * 0.2]
        
    for i in range(num_to_anneal):
        if _remaining_time() < 5.0:
            break
            
        e, c = candidates_with_energy[i]
        budget = min(budgets[i], _remaining_time() - 3.0)
        if budget < 0.5:
            break
            
        annealed = _anneal_reversal(sequence, c, budget, rng)
        e_annealed = _energy(sequence, annealed)
        
        if e_annealed < best_energy:
            best_energy = e_annealed
            best_chain = annealed
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
