import time
import random
import math

N = 5000
DEADLINE = 325.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p] and p <= limit]

def _initial_greedy():
    """
    Fast deterministic heuristic to generate a valid starting tuple.
    Starts with all points and removes least populated residue classes.
    """
    points = set(range(N + 1))
    primes = _primes(N + 1)
    
    for p in primes:
        if p > len(points):
            break
        
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        if min(counts) > 0:
            min_val = min(counts)
            min_classes = [i for i, c in enumerate(counts) if c == min_val]
            # Remove all points in the least populated class(es)
            # Actually, just removing one class is enough to satisfy 'missed', 
            # but removing all min-class points maximizes size reduction? No, 
            # we want to REMOVE as few as possible. 
            # Wait, the constraint is: some residue class is missed.
            # If min(counts) > 0, NO class is missed. We MUST remove points.
            # To maximize remaining size, we should remove the class with the FEWEST points.
            # So we pick ONE class with min count and remove it.
            target_class = random.choice(min_classes)
            points = {v for v in points if v % p != target_class}
            
    return list(points)

def construct():
    start_time = time.time()
    
    # 1. Get initial valid solution
    current_points = _initial_greedy()
    in_set = set(current_points)
    current_size = len(current_points)
    best_size = current_size
    best_points = list(current_points)
    
    # 2. Precompute primes
    # We need primes up to N+1 potentially, but constraints only apply to p <= size.
    # As size changes, the set of relevant primes changes.
    # Max possible size is N+1.
    primes = _primes(N + 1)
    
    # 3. Simulated Annealing Setup
    # State: set of points
    # Move: 
    #   - Add a point from [0, N] not in set
    #   - Remove a point from set
    #   - Swap a point in set with a point not in set
    # Objective: Maximize size, subject to admissibility.
    # Penalty: If not admissible, large penalty.
    
    # To speed up evaluation, we maintain counts for relevant primes.
    # However, 'relevant primes' depend on current size.
    # Let's precompute residue classes for all points for all primes?
    # N=5000, pi(5000)~669. 5000 * 669 ~ 3.3M integers. Memory ~ 25MB. Acceptable.
    # But updating counts for every move is O(1) if we have precomputed residues.
    
    # Precompute residues: residues[p][v] = v % p
    # This is too much memory if stored as list of lists of lists.
    # Better: Just compute v % p on the fly? 
    # 300s budget. 
    # Let's estimate operations.
    # If we do 100k moves, and each move checks ~100 primes (primes <= 5000 is 669, but we only care about p <= size ~ 2000? No, size can be up to 5000).
    # Actually, if size is ~2000, we check primes <= 2000.
    # 100k moves * 300 primes * (add/remove cost)
    # Add/remove cost is O(1) update to counts.
    # Check cost is O(num_primes).
    # 100k * 300 = 30M ops. Python can do ~10-50M simple ops/sec.
    # So 30M ops might take 1-3 seconds. This is very fast.
    # We can afford millions of moves.
    
    # Let's use a simpler representation:
    # in_set: set of ints
    # counts: dict p -> list of counts
    # missed: dict p -> set of missed residues
    
    # Initialize counts and missed for current state
    counts = {}
    missed = {}
    
    def update_state_add(v):
        """Update counts/missed when adding v. Returns True if valid move in terms of constraints? No, just updates stats."""
        for p in primes:
            if p > len(in_set) + 1: # We might need p = len+1 for future checks, but for current validity p <= len
                 # Actually, we need to check validity AFTER the move.
                 # So we need stats for all p <= new_len.
                 # If we are adding, new_len = old_len + 1.
                 # We need stats for p <= new_len.
                 pass
            r = v % p
            if p not in counts:
                counts[p] = [0] * p
                missed[p] = set(range(p))
            counts[p][r] += 1
            if r in missed[p]:
                missed[p].discard(r)

    def update_state_remove(v):
        for p in primes:
            if p not in counts:
                continue
            r = v % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                missed[p].add(r)

    # Initialize stats for current best_points
    for v in best_points:
        update_state_add(v)
        
    # Note: The above loop initializes counts for ALL primes in `primes` list?
    # No, `update_state_add` iterates `primes`. 
    # Wait, `primes` is up to N+1. 
    # This initialization is O(N * pi(N)) ~ 3M ops. Fast.
    
    # However, `update_state_add` creates `counts[p]` if not present.
    # Since we iterate all primes in `primes` list, all will be created.
    
    # Helper to check admissibility
    def is_admissible(size):
        for p in primes:
            if p > size:
                break
            if len(missed[p]) == 0:
                return False
        return True

    # Helper to evaluate a move (add/remove/swap)
    # We will implement a "trial" mechanism that reverts if we decide to reject.
    
    T = 10.0
    T_min = 0.001
    alpha = 0.999 # Cooling factor
    
    # We need to track the set of available candidates (not in set)
    # To make "Add" moves efficient, we can just pick a random number from 0..N and check if in set.
    # Or maintain a list of candidates.
    # Given N=5000, picking random and checking set is O(1).
    
    moves_made = 0
    while time.time() - start_time < DEADLINE - 5:
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # If T is very low, we are effectively doing greedy/local search
            # We can break or continue with greedy.
            # Let's continue to polish.
            
        # Choose a move
        # 1. Add a point
        # 2. Remove a point
        # 3. Swap
        
        move_type = random.random()
        
        old_size = len(in_set)
        old_admissible = is_admissible(old_size)
        
        # If current state is invalid, we MUST fix it.
        # Penalty approach:
        # Score = size - M * (number of violations)
        # Violation = sum over p <= size of (1 if missed[p] is empty else 0)
        
        def get_violations(size):
            v_count = 0
            for p in primes:
                if p > size:
                    break
                if len(missed[p]) == 0:
                    v_count += 1
            return v_count

        current_violations = get_violations(old_size)
        current_score = old_size - 1000 * current_violations
        
        new_score = current_score
        move_executed = False
        undo_ops = [] # List of (type, val) to revert
        
        if move_type < 0.6:
            # Try Add
            # Pick random candidate
            cand = random.randint(0, N)
            if cand in in_set:
                # Try a few times to find a candidate not in set
                for _ in range(10):
                    cand = random.randint(0, N)
                    if cand not in in_set:
                        break
                else:
                    # If set is full, we can't add.
                    # This shouldn't happen often as we remove too.
                    continue
            
            # Execute Add
            in_set.add(cand)
            update_state_add(cand)
            undo_ops.append(('add', cand))
            
            new_size = old_size + 1
            new_violations = get_violations(new_size)
            new_score = new_size - 1000 * new_violations
            
            move_executed = True
            
        elif move_type < 0.8:
            # Try Remove
            if not in_set:
                continue
            cand = random.choice(list(in_set))
            
            # Execute Remove
            in_set.remove(cand)
            update_state_remove(cand)
            undo_ops.append(('remove', cand))
            
            new_size = old_size - 1
            new_violations = get_violations(new_size)
            new_score = new_size - 1000 * new_violations
            
            move_executed = True
            
        else:
            # Try Swap
            if not in_set:
                continue
            cand_in = random.choice(list(in_set))
            cand_out = random.randint(0, N)
            if cand_out in in_set:
                # Try to find distinct
                for _ in range(10):
                    cand_out = random.randint(0, N)
                    if cand_out not in in_set:
                        break
                else:
                    continue
            
            # Execute Swap: Remove cand_in, Add cand_out
            in_set.remove(cand_in)
            update_state_remove(cand_in)
            in_set.add(cand_out)
            update_state_add(cand_out)
            undo_ops.append(('swap', cand_in, cand_out))
            
            new_size = old_size
            new_violations = get_violations(new_size)
            new_score = new_size - 1000 * new_violations
            
            move_executed = True
        
        if not move_executed:
            continue
            
        delta = new_score - current_score
        
        # Acceptance
        accept = False
        if delta > 0:
            accept = True
        else:
            if T > 0:
                # Avoid log(0) if delta is 0
                if delta == 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(delta / T)
                        accept = random.random() < prob
                    except OverflowError:
                        accept = False
        
        if accept:
            # Keep move
            moves_made += 1
            # Update best if valid and larger
            if new_violations == 0 and new_size > best_size:
                best_size = new_size
                best_points = list(in_set)
        else:
            # Revert
            for op in reversed(undo_ops):
                if op[0] == 'add':
                    # Undo add: remove
                    v = op[1]
                    in_set.remove(v)
                    update_state_remove(v)
                elif op[0] == 'remove':
                    # Undo remove: add
                    v = op[1]
                    in_set.add(v)
                    update_state_add(v)
                elif op[0] == 'swap':
                    # Undo swap: remove cand_out, add cand_in
                    v_in, v_out = op[1], op[2]
                    in_set.remove(v_out)
                    update_state_remove(v_out)
                    in_set.add(v_in)
                    update_state_add(v_in)

    # Return the best valid solution found
    return best_points

# EVOLVE-BLOCK-END
