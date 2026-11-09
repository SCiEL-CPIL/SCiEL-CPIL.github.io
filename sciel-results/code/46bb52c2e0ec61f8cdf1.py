<<<<<<< SEARCH
def _backtrack_with_contacts(sequence, dimension, target_contacts, rng, time_limit):
    """
    Backtracking search that prioritizes realizing target contacts.
    At each step, prefer moves that create desired H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    realized_contacts = set()
    
    start_time = time.time()
    max_depth = n
    
    def get_heuristic_score(nb, i):
        """Score a candidate position based on contact potential."""
        score = 0.0
        
        if sequence[i] == 'H':
            # Check if placing H at nb creates any target contacts
            for j in range(i):
                if sequence[j] == 'H':
                    # Check if nb is adjacent to chain[j]
                    dist = abs(nb[0]-chain[j][0]) + abs(nb[1]-chain[j][1]) + abs(nb[2]-chain[j][2])
                    if dist == 1:
                        # Is (j, i) in target contacts?
                        if (j, i) in target_contacts or (i, j) in target_contacts:
                            score += 100  # Strong preference for target contacts
                        else:
                            score += 10   # Some preference for any H-H contact
        
        # Compactness: prefer positions closer to the center of mass
        if len(chain) > 1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist_c = (nb[0]-cx)**2 + (nb[1]-cy)**2 + (nb[2]-cz)**2
            score -= dist_c * 0.1
        
        return score
    
    def backtrack(i):
        if i == n:
            return True
        
        if time.time() - start_time > time_limit * 0.5:
            return False  # Time out
        
        last = chain[-1]
        neighbors = list(_get_neighbors(last))
        rng.shuffle(neighbors)
        
        # Sort by heuristic
        neighbors.sort(key=lambda nb: get_heuristic_score(nb, i), reverse=True)
        
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied[nb] = i
                
                # Check if this creates any new target contacts
                if sequence[i] == 'H':
                    for j in range(i):
                        if sequence[j] == 'H':
                            dist = abs(nb[0]-chain[j][0]) + abs(nb[1]-chain[j][1]) + abs(nb[2]-chain[j][2])
                            if dist == 1 and ((j, i) in target_contacts or (i, j) in target_contacts):
                                realized_contacts.add((min(j,i), max(j,i)))
                
                if backtrack(i + 1):
                    return True
                
                chain.pop()
                del occupied[nb]
        
        return False
    
    if not backtrack(1):
        # Fallback: generate a valid chain using greedy
        return _greedy_fold(sequence, dimension, rng)
    
    return chain
=======
def _backtrack_with_contacts(sequence, dimension, target_contacts, rng, time_limit):
    """
    Backtracking search with efficient contact tracking.
    Uses a neighbor set for O(1) contact detection.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    h_neighbors = defaultdict(set)  # For each H index, set of neighboring H indices
    
    start_time = time.time()
    
    def get_heuristic_score(nb, i):
        """Score a candidate position based on contact potential."""
        score = 0.0
        
        if sequence[i] == 'H':
            # Check neighbors of nb for H residues
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                neighbor_pos = (nb[0]+dx, nb[1]+dy, nb[2]+dz)
                if neighbor_pos in occupied:
                    j = occupied[neighbor_pos]
                    if sequence[j] == 'H' and j < i:
                        if (min(j,i), max(j,i)) in target_contacts:
                            score += 50
                        else:
                            score += 10
        
        # Compactness: prefer positions closer to the center of mass
        if len(chain) > 1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist_c = (nb[0]-cx)**2 + (nb[1]-cy)**2 + (nb[2]-cz)**2
            score -= dist_c * 0.05
        
        return score
    
    def backtrack(i):
        if i == n:
            return True
        
        if time.time() - start_time > time_limit:
            return False  # Time out
        
        last = chain[-1]
        neighbors = list(_get_neighbors(last))
        
        # Filter valid neighbors and score them
        valid_neighbors = []
        for nb in neighbors:
            if nb not in occupied:
                valid_neighbors.append((get_heuristic_score(nb, i), nb))
        
        # Sort by score (higher is better)
        valid_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        for _, nb in valid_neighbors:
            chain.append(nb)
            occupied[nb] = i
            
            # Update contact tracking
            if sequence[i] == 'H':
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    neighbor_pos = (nb[0]+dx, nb[1]+dy, nb[2]+dz)
                    if neighbor_pos in occupied:
                        j = occupied[neighbor_pos]
                        if sequence[j] == 'H':
                            h_neighbors[i].add(j)
                            h_neighbors[j].add(i)
            
            if backtrack(i + 1):
                return True
            
            chain.pop()
            del occupied[nb]
            
            # Undo contact tracking
            if sequence[i] == 'H':
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    neighbor_pos = (nb[0]+dx, nb[1]+dy, nb[2]+dz)
                    if neighbor_pos in occupied:
                        j = occupied[neighbor_pos]
                        if sequence[j] == 'H':
                            h_neighbors[i].discard(j)
                            h_neighbors[j].discard(i)
        
        return False
    
    if not backtrack(1):
        # Fallback: generate a valid chain using greedy
        return _greedy_fold(sequence, dimension, rng)
    
    return chain
>>>>>>> REPLACE

