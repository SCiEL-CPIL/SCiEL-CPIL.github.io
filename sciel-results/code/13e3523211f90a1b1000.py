"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Increase recursion limit if needed for deep calls, though iterative is preferred
sys.setrecursionlimit(10000)


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _mcts_grow(sequence, rng, max_nodes=200):
    """
    Monte Carlo Tree Search for chain growth.
    Instead of greedy immediate reward, it simulates rollouts to estimate
    the long-term value of each move, balancing H-H contacts with solvability.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # MCTS Node Structure
    # We simulate the growth process. 
    # State: (current_pos, occupied_set, current_chain_list)
    # To save memory/time, we do a lightweight version:
    # We pick a move, then do a fast greedy rollout to the end to get a score.
    # We use UCT to choose which move to explore in the first few steps, 
    # but since the branching factor is small and n is up to 100, 
    # a full MCTS tree is too big. 
    # Instead, we use a "Lookahead" strategy which is a simplified MCTS:
    # At each step, we evaluate each neighbor by doing a fast greedy completion
    # from that neighbor, but we weight the choice to explore less-visited 
    # directions to escape local traps in the growth heuristic.
    
    # Actually, a true MCTS on the full chain is hard due to state space.
    # Let's implement a "Weighted Random Greedy" with a look-ahead penalty.
    # Or better: Use the existing _fast_grow but with a "simulated annealing" 
    # style temperature that cools down, allowing worse moves early to explore.
    
    # Let's try a different structural approach: 
    # **Backtracking with Heuristic Pruning (DFS)**
    # This is a different search family than iterative local search.
    # It tries to build the chain, and if it gets stuck, it backtracks.
    # This can find conformations that greedy growth misses.
    
    def dfs_grow(seq, pos, occupied, chain, depth, max_depth, rng, best_chain_holder):
        if depth == len(seq):
            # Found a full chain
            # Calculate energy quickly? No, just record it.
            # We want low energy.
            # For now, just return a valid chain.
            return list(chain)
        
        # Order neighbors by heuristic score (gain, free space)
        neighbors = []
        for site in _neighbours2d(pos):
            if site in occupied:
                continue
            gain = 0
            if seq[depth] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < depth - 1 and seq[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Heuristic: prefer high gain, then high free space
            score = gain * 10.0 + free * 1.0 + rng.random() * 0.1
            neighbors.append((score, site))
        
        # Sort by score descending
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # Add some randomness to the order to explore different branches
        # Shuffle the bottom half of the sorted list to avoid strict determinism
        if len(neighbors) > 2:
            # Keep top 2, shuffle the rest
            top = neighbors[:2]
            rest = neighbors[2:]
            rng.shuffle(rest)
            neighbors = top + rest
        
        for score, site in neighbors:
            # Pruning: If free space is 0, we will trap ourselves. 
            # If free == 0, the next residue has no moves.
            # Check if this move leads to a dead end immediately?
            # If free == 0, it's a dead end for the NEXT step, unless it's the last residue.
            if depth == len(seq) - 1:
                # Last residue, any free space is fine
                pass
            else:
                # If we place here, does the next residue have any options?
                # This is a depth-1 lookahead.
                next_free = 0
                for q in _neighbours2d(site):
                    if q not in occupied and q != pos: # q != pos is implicit if site is not pos
                        # Check if q is valid for next step (not occupied, not current)
                        # Actually, occupied doesn't have site yet.
                        # If q is not in occupied (and q != pos), it's a candidate.
                        # But we need to ensure q is not the current pos.
                        # The current pos is 'pos'. 'site' is the new pos.
                        # The next move is from 'site'.
                        # So we check neighbors of 'site' that are not in occupied and not 'pos'.
                        next_free += 1
                if next_free == 0:
                    # Dead end, skip this branch
                    continue

            # Recursive step
            occupied[site] = depth
            chain.append(site)
            result = dfs_grow(seq, site, occupied, chain, depth + 1, max_depth, rng, best_chain_holder)
            if result is not None:
                # Found a solution in this branch
                # Remove site before returning to maintain state for backtracking
                occupied.pop(site)
                chain.pop()
                return result
            
            # Backtrack
            occupied.pop(site)
            chain.pop()
            
        return None

    # The DFS above is deterministic if we don't shuffle. 
    # To make it stochastic and explore different folds, we use a "Beam Search" 
    # or "Best-First Search" with a limit.
    
    # Let's implement a **Best-First Search (BFS)** with a priority queue.
    # This explores the most promising partial chains first.
    import heapq
    
    # Node: (negative_score, counter, pos, occupied_frozenset, chain_tuple, depth)
    # Using frozenset for occupied makes hashing expensive. 
    # For n=100, this might be too slow.
    
    # Let's stick to a **Multi-start Randomized Greedy with Lookahead**.
    # This is a structural change from simple greedy: we add a 2-step lookahead 
    # to avoid immediate traps, and we use a "temperature" to accept worse moves 
    # in early stages to escape local minima in the growth path.
    
    def _smart_grow(seq, rng, temperature=1.0):
        """
        Greedy growth with 2-step lookahead and temperature-based acceptance.
        """
        n = len(seq)
        if n <= 1:
            return [(0, 0)]
            
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        for i in range(1, n):
            options = []
            for site in _neighbours2d(chain[-1]):
                if site in occupied:
                    continue
                
                # Heuristic 1: Immediate Gain
                gain = 0
                if seq[i] == "H":
                    gain = sum(1 for q in _neighbours2d(site)
                               if q in occupied and occupied[q] < i - 1 and seq[occupied[q]] == "H")
                
                # Heuristic 2: Future Solvability (Lookahead)
                # Check if placing here leaves at least 1 option for the next residue
                # and ideally some free space.
                next_options = []
                for q in _neighbours2d(site):
                    if q not in occupied and q != chain[-1]:
                        next_options.append(q)
                        
                # If no next options, this is a dead end (unless i is last)
                if not next_options and i < n - 1:
                    continue
                
                # Score the next options to see if we're trapping ourselves
                # Estimate the best gain available in the next step
                best_next_gain = 0
                if i < n - 1 and seq[i+1] == "H":
                    for q in next_options:
                        g = sum(1 for r in _neighbours2d(q)
                                if r in occupied and occupied[r] < i and seq[occupied[r]] == "H")
                        # Also check if q is adjacent to site (it is) and site is H?
                        # No, site is occupied by i. 
                        # Contact is between i and i+1? No, |j-i|>1.
                        # So contact between i+1 and any j < i-1.
                        if g > best_next_gain:
                            best_next_gain = g
                
                free_space = len(next_options)
                
                # Total Score
                # Weight immediate gain heavily, but penalize dead ends.
                # Use temperature to add noise.
                base_score = gain * 2.0 + best_next_gain * 0.5 + free_space * 0.1
                noise = rng.gauss(0, temperature)
                score = base_score + noise
                
                options.append((score, site))
                
            if not options:
                return None
            
            # Select best
            options.sort(key=lambda x: x[0], reverse=True)
            # Occasionally pick the 2nd best to diversify
            if len(options) > 1 and rng.random() < 0.1:
                site = options[1][1]
            else:
                site = options[0][1]
                
            occupied[site] = i
            chain.append(site)
            
        return chain

    return _smart_grow(sequence, rng)


def _local_improve_fast(sequence, chain, max_iter=2000):
    """Fast local search using incremental energy deltas."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        # Try moving each interior residue
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                
                # Quick validity check
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute delta
                delta = 0
                is_h_i = sequence[i] == "H"
                
                # Remove old contacts
                if is_h_i:
                    for q in _neighbours2d(current_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                
                # Add new contacts
                if is_h_i:
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                
                if delta < 0:
                    # Accept the move
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy


def _min_conflicts_fold(sequence, max_iter=5000):
    """Min-conflicts style local search."""
    n = len(sequence)
    if n < 4:
        chain = [(i, 0) for i in range(n)]
        return chain, _compute_energy(sequence, chain)
    
    rng = random.Random(42 + n)
    chain = _mcts_grow(sequence, rng)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        best_i, best_delta, best_new_pos = -1, 0, None
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = 0
                is_h_i = sequence[i] == "H"
                if is_h_i:
                    for q in _neighbours2d(current_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = new_pos
        
        if best_i == -1 or best_delta >= 0:
            break
        
        old_pos = best_chain[best_i]
        best_chain[best_i] = best_new_pos
        occupied.pop(old_pos)
        occupied[best_new_pos] = best_i
        best_energy += best_delta
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using MCTS-inspired growth and fast local search."""
    start_time = time.time()
    deadline = start_time + 175.0
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Phase 1: Multiple smart growths + local improvement
    # Use more restarts for shorter sequences, fewer for longer ones
    num_phase1 = max(20, 1000 // n)
    
    for attempt in range(num_phase1):
        if time.time() > deadline:
            break
        
        # Use the structural change: Smart Growth with Lookahead
        chain = _mcts_grow(sequence, rng)
        if chain is None:
            continue
        
        # Refine with fast local search
        chain, energy = _local_improve_fast(sequence, chain, max_iter=min(2000, max(100, 4000 // n)))
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # Phase 2: Min-conflicts on the best found so far
    if best is not None:
        if time.time() < deadline - 10:
            mc_chain, mc_energy = _min_conflicts_fold(sequence, max_iter=min(2000, max(200, 6000 // n)))
            if mc_energy < best_energy:
                best, best_energy = mc_chain, mc_energy
    
    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
