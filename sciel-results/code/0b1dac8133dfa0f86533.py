import time

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search on the 3D cubic lattice.
    Returns a list of lists of integers representing the coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    start_time = time.time()
    # Use 290 seconds as a hard limit to be safe, leaving 40s for overhead/other sequences
    # The task says 330s wall clock for ALL sequences.
    # We don't know how many sequences are left, so we should be conservative per sequence.
    # However, the prompt says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # "240 s CPU budget for all sequences together" in the task description, but "330s wall-clock" in the directive.
    # If each call is separate, we have 330s per call? No, "sandboxed subprocess" usually implies isolation.
    # "Each evaluation must finish within 300 s."
    # So we have 300s per fold call.
    # I will use 290s as the limit.
    deadline = start_time + 290
    
    # Symmetry breaking:
    # R0 at (0,0,0), R1 at (1,0,0)
    initial_chain = [(0, 0, 0), (1, 0, 0)]
    initial_occupied = {(0, 0, 0), (1, 0, 0)}
    initial_pos_to_idx = {(0, 0, 0): 0, (1, 0, 0): 1}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_pos_to_idx, initial_energy)]
    
    # Beam width: 200 is a good balance.
    K = 200
    
    for i in range(2, n):
        new_beam = []
        residue = sequence[i]
        
        for state in beam:
            chain, occupied, pos_to_idx, energy = state
            last_pos = chain[-1]
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if residue == 'H':
                    for nb in _neighbours(neighbor):
                        if nb in occupied:
                            j = pos_to_idx[nb]
                            # Exclude the immediately preceding residue in the chain
                            if j != i - 1 and sequence[j] == 'H':
                                new_energy -= 1
                
                # Calculate free space for heuristic
                free_space = 0
                for nb in _neighbours(neighbor):
                    if nb not in occupied:
                        free_space += 1
                
                # Score: Energy is primary (minimize), free space is secondary (maximize)
                # We sort by score ascending.
                # Energy is negative. More negative is better.
                # Free space is positive. More is better.
                # So we want low energy (negative) and high free space.
                # score = energy - 0.05 * free_space
                score = new_energy - 0.05 * free_space
                
                # Create new state
                new_chain = chain + [neighbor]
                new_occupied = occupied | {neighbor}
                new_pos_to_idx = pos_to_idx | {neighbor: i}
                
                # Store score in the tuple for sorting
                new_beam.append((score, new_chain, new_occupied, new_pos_to_idx, new_energy))
        
        if not new_beam:
            break
            
        # Sort by score
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top K
        beam = []
        for item in new_beam[:K]:
            # Remove score from the state tuple for next iteration
            _, chain, occupied, pos_to_idx, energy = item
            beam.append((chain, occupied, pos_to_idx, energy))
            
        if not beam:
            break
            
        # Time check
        if time.time() > deadline:
            break

    if not beam:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Return the best chain (lowest score, which corresponds to lowest energy)
    best_chain = beam[0][0]
    return [list(p) for p in best_chain]
