# EVOLVE-BLOCK-START
import time
import sys
import math
import random
from heapq import heappush, heappop

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, beam_width=100, time_limit=2.0):
    """
    Beam search that builds the chain from the N-terminus.
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start = (0, 0, 0)
    initial_chain = (start,)
    initial_occupied = {start}
    initial_energy = 0
    
    # State: (energy, chain, occupied)
    # We use a list, sorted by energy at each step.
    beam = [(initial_energy, initial_chain, initial_occupied)]
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() - start_time > time_limit:
            break
            
        new_beam = []
        current_residue = sequence[step]
        
        for energy, chain, occupied in beam:
            last = chain[-1]
            
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                new_energy = energy
                if current_residue == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Find index of q in chain
                            # Since chain is a tuple, we can't use .index efficiently if it's long?
                            # Actually, we need to check if the residue at q is H.
                            # We need a mapping from position to index.
                            # To avoid O(N) lookup in the inner loop, we should store the mapping in the state.
                            # Let's change state to (energy, chain, occupied, pos_to_idx)
                            # But copying pos_to_idx is expensive.
                            
                            # Alternative: Just do a linear scan.
                            # N is up to 136. Beam width 100.
                            # 100 * 6 neighbors * 136 scan = 81,600 ops per step.
                            # 136 steps * 81,600 = 11M ops. This is okay in Python if optimized?
                            # Let's try to be smarter.
                            
                            # We can precompute a list of H positions for the current chain?
                            # No, that's also O(N).
                            
                            # Let's stick to linear scan but optimize it.
                            for j, p in enumerate(chain):
                                if p == q:
                                    if sequence[j] == "H":
                                        new_energy -= 1
                                    break
                            
                new_chain = chain + (next_pos,)
                new_occupied = occupied | {next_pos}
                new_beam.append((new_energy, new_chain, new_occupied))
                
        if not new_beam:
            break
            
        # Sort by energy (ascending, so most negative first)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
    if beam:
        return list(beam[0][1])
    return None

def _reverse_beam_search_fold(sequence, dimension, beam_width=100, time_limit=2.0):
    """
    Beam search that builds the chain from the C-terminus backwards.
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    # Reverse the sequence
    rev_seq = sequence[::-1]
    
    start = (0, 0, 0)
    initial_chain = (start,)
    initial_occupied = {start}
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied)]
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() - start_time > time_limit:
            break
            
        new_beam = []
        current_residue = rev_seq[step]
        
        for energy, chain, occupied in beam:
            last = chain[-1]
            
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                new_energy = energy
                if current_residue == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            for j, p in enumerate(chain):
                                if p == q:
                                    if rev_seq[j] == "H":
                                        new_energy -= 1
                                    break
                            
                new_chain = chain + (next_pos,)
                new_occupied = occupied | {next_pos}
                new_beam.append((new_energy, new_chain, new_occupied))
                
        if not new_beam:
            break
            
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
    if beam:
        # Reverse the chain back
        return list(reversed(beam[0][1]))
    return None

def _local_optimize(sequence, chain, dimension, time_limit=2.0):
    """
    Local optimization using pivot moves (translation of tail).
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute occupied set for fast lookup
    # We will maintain a set of occupied positions for the "head" part (0..i)
    
    while time.time() - start_time < time_limit:
        # Pick a random pivot
        i = rng.randint(1, n - 2)
        
        prev = chain[i]
        curr = chain[i+1]
        
        valid_moves = []
        for q in _neighbours(prev):
            if q == curr:
                continue
            
            # Check if q is occupied by any part of the chain EXCEPT the tail i+1..n-1
            # We need to be careful.
            # Let's build a set of occupied positions for indices 0..i
            # This is O(N) per pivot selection, which is slow.
            # Optimization: Maintain a global occupied set?
            # If we do, we need to remove the tail and add it back.
            
            # Let's try a simpler check:
            # If q is in chain[0..i], it's invalid.
            # We can do this with a set built once per iteration? No, that's O(N).
            
            # Let's just iterate chain[0..i]
            is_valid = True
            for j in range(i + 1):
                if chain[j] == q:
                    is_valid = False
                    break
            
            if is_valid:
                valid_moves.append(q)
        
        if not valid_moves:
            continue
            
        new_curr = rng.choice(valid_moves)
        
        delta = tuple(new_curr[k] - curr[k] for k in range(dimension))
        
        new_chain = chain[:i+1] + [new_curr]
        
        # Check collisions for the rest of the tail
        # We need to check against chain[0..i] and the new tail itself.
        # Let's build a set of occupied for chain[0..i]
        head_occupied = set(chain[:i+1])
        
        success = True
        for j in range(i+2, n):
            old_p = chain[j]
            new_p = tuple(old_p[k] + delta[k] for k in range(dimension))
            
            if new_p in head_occupied:
                success = False
                break
            
            # Check against already placed tail parts
            # This is tricky. We need to check against new_chain[i+1:]
            # Since we are building new_chain, we can just check against the set of placed tail parts.
            # Let's maintain a set `tail_occupied`
            
            # Actually, the simple check `if new_p in new_chain` is O(N).
            # Let's use a set.
            pass
        
        # Let's re-implement the collision check properly with a set.
        # This is getting complex and potentially slow.
        # Let's use a simpler local search: random walk with acceptance based on energy.
        # Or just use the pivot move with a full energy calculation if accepted?
        # No, we need to check validity first.
        
        # Let's try a different local search: "Slither" move?
        # Or just random perturbation of a few residues?
        
        # Given time constraints, let's stick to a simple random restart strategy in the main fold function
        # and skip complex local optimization if it's too slow to implement correctly.
        
        # Let's try a very simple local move: swap two non-adjacent residues? No, that breaks connectivity.
        
        # Let's just do a random walk on the lattice?
        # No, let's just use the beam search results directly.
        # The beam search itself is a form of local optimization (greedy with look-ahead).
        
        # If we have time, we can do a simple "kick" move:
        # Pick a random index k, try to move chain[k] to a neighbor if it doesn't collide.
        # But this breaks the chain unless we move the whole tail.
        
        # Let's implement the pivot move correctly with a set.
        
        head_occupied = set(chain[:i+1])
        tail_occupied = set()
        
        success = True
        new_tail = []
        for j in range(i+1, n):
            old_p = chain[j]
            if j == i+1:
                new_p = new_curr
            else:
                new_p = tuple(old_p[k] + delta[k] for k in range(dimension))
            
            if new_p in head_occupied or new_p in tail_occupied:
                success = False
                break
                
            tail_occupied.add(new_p)
            new_tail.append(new_p)
            
        if not success:
            continue
            
        new_chain = chain[:i+1] + new_tail
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            
    return best_chain

def fold(sequence, dimension):
    """
    Uses multiple beam search strategies (forward, backward) and picks the best.
    """
    n = len(sequence)
    
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 300.0
    current_time = time.time()
    
    if current_time >= global_deadline:
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Budget for this call
    # We have 26 sequences.
    # Let's allocate time based on remaining budget.
    # Assume we are called sequentially.
    
    # Simple heuristic: give more time to longer chains?
    # Or just split the remaining time equally?
    # We don't know how many calls are left.
    # Let's assume a fixed budget per call, capped by remaining global time.
    
    if n < 50:
        call_time = 5.0
    elif n < 100:
        call_time = 8.0
    else:
        call_time = 12.0
        
    call_deadline = current_time + call_time
    if call_deadline > global_deadline:
        call_deadline = global_deadline
        
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Forward Beam Search
    time_per_search = (call_deadline - current_time) / 3.0
    if time_per_search > 0.5:
        chain1 = _beam_search_fold(sequence, dimension, beam_width=100, time_limit=time_per_search)
        if chain1:
            e1 = _energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
                
    # Strategy 2: Backward Beam Search
    if time.time() < call_deadline - 0.5:
        time_per_search = (call_deadline - time.time()) / 2.0
        if time_per_search > 0.5:
            chain2 = _reverse_beam_search_fold(sequence, dimension, beam_width=100, time_limit=time_per_search)
            if chain2:
                e2 = _energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
                    
    # Strategy 3: Local Optimization on best found
    if best_chain and time.time() < call_deadline - 0.5:
        opt_time = max(0.5, call_deadline - time.time() - 0.5)
        optimized = _local_optimize(sequence, best_chain, dimension, time_limit=opt_time)
        opt_energy = _energy(sequence, optimized)
        
        if opt_energy < best_energy:
            best_chain = optimized
            best_energy = opt_energy
            
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
