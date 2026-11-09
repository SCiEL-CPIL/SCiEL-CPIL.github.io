import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters
    N = q ** n
    deadline = time.time() + 110.0
    
    # Precompute balls
    # idx to digits
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    balls = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
        
    # Initial code: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
            
    best_size = len(initial_codewords)
    best_code = initial_codewords[:]
    
    # Simulated Annealing Parameters
    M = 1000 # Penalty for uncovered words
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9999
    
    # Run multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize state
        # For restarts > 0, we can perturb the initial code
        current_code = initial_codewords[:]
        if r > 0:
            # Randomly swap a few words to break symmetry
            for _ in range(5):
                if len(current_code) > 0:
                    i = random.randint(0, len(current_code) - 1)
                    new_word = random.randint(0, N - 1)
                    # Only swap if it doesn't create a duplicate (check simple)
                    if new_word not in current_code:
                        current_code[i] = new_word
        
        # Setup coverage
        cov = [0] * N
        in_code = [False] * N
        for c in current_code:
            in_code[c] = True
            for v in balls[c]:
                cov[v] += 1
        
        U = sum(1 for c in cov if c == 0)
        E = len(current_code) + M * U
        
        T = T_start
        no_improve_steps = 0
        local_best_size = len(current_code)
        
        steps = 0
        while T > T_end and time.time() < deadline - 1.0:
            steps += 1
            # Generate move
            move_type = random.random()
            delta_E = 0
            accepted = False
            
            # We need to calculate delta_E for the move
            # Move: Delete c
            if move_type < 0.3 and len(current_code) > 0:
                i = random.randint(0, len(current_code) - 1)
                c = current_code[i]
                
                # Calculate change in U
                # Removing c decreases cov[v] by 1 for v in balls[c]
                # U increases if cov[v] becomes 0
                delta_U = 0
                for v in balls[c]:
                    if cov[v] == 1:
                        delta_U += 1
                
                delta_size = -1
                delta_E = delta_size + M * delta_U
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    accepted = True
                    # Apply move
                    in_code[c] = False
                    for v in balls[c]:
                        cov[v] -= 1
                    current_code.pop(i)
                    U += delta_U
                    E += delta_E
                    
                    if U == 0 and len(current_code) < local_best_size:
                        local_best_size = len(current_code)
                        no_improve_steps = 0
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = current_code[:]
                    else:
                        no_improve_steps += 1
                        
            # Move: Add v
            elif move_type < 0.5:
                v = random.randint(0, N - 1)
                if not in_code[v]:
                    # Calculate change in U
                    # Adding v increases cov[v] by 1 for v in balls[v]
                    # U decreases if cov[v] was 0
                    delta_U = 0
                    for v2 in balls[v]:
                        if cov[v2] == 0:
                            delta_U -= 1
                    
                    delta_size = 1
                    delta_E = delta_size + M * delta_U
                    
                    if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                        # Accept
                        accepted = True
                        # Apply move
                        in_code[v] = True
                        for v2 in balls[v]:
                            cov[v2] += 1
                        current_code.append(v)
                        U += delta_U
                        E += delta_E
                        
                        if U == 0 and len(current_code) < local_best_size:
                            local_best_size = len(current_code)
                            no_improve_steps = 0
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code = current_code[:]
                        else:
                            no_improve_steps += 1
            else:
                # Move: Swap c with v
                if len(current_code) > 0:
                    i = random.randint(0, len(current_code) - 1)
                    c = current_code[i]
                    v = random.randint(0, N - 1)
                    if v != c and not in_code[v]:
                        # Remove c, Add v
                        # Delta U is sum of changes
                        delta_U = 0
                        # Remove c
                        for v2 in balls[c]:
                            if cov[v2] == 1:
                                delta_U += 1
                        # Add v
                        # Note: cov is not updated yet.
                        # We need to be careful if balls[c] and balls[v] overlap.
                        # If they overlap, the net change is more complex.
                        # Let's just calculate the final U change by simulating on a copy?
                        # No, that's slow.
                        # Let's count:
                        # For each v2 in union(balls[c], balls[v]):
                        #   old_cov = cov[v2]
                        #   new_cov = old_cov - (1 if v2 in balls[c] else 0) + (1 if v2 in balls[v] else 0)
                        #   if old_cov == 0 and new_cov > 0: delta_U -= 1
                        #   if old_cov > 0 and new_cov == 0: delta_U += 1
                        
                        # This is O(26) per move. Acceptable.
                        # But we need to know membership in balls.
                        # balls are lists. Membership check is O(13).
                        # So O(26 * 13) ~ 300 ops.
                        
                        # Alternative: Just apply, calc U, revert if reject.
                        # Applying is O(13+13).
                        # Calc U from scratch is O(729).
                        # This is too slow for inner loop.
                        
                        # Let's stick to the incremental delta_U calculation.
                        # We need to iterate over unique vertices in balls[c] U balls[v].
                        # Since balls are small, we can just do:
                        
                        # 1. Temporarily remove c from cov
                        for v2 in balls[c]:
                            cov[v2] -= 1
                        # 2. Temporarily add v to cov
                        for v2 in balls[v]:
                            cov[v2] += 1
                        
                        # 3. Calculate delta_U
                        # We need to know how many 0->1 and 1->0 transitions happened.
                        # This is hard to do without storing old state.
                        
                        # Let's revert the temp changes and use a different strategy.
                        # Strategy:
                        # 1. Store old cov values for affected vertices.
                        # 2. Apply changes.
                        # 3. Count U change.
                        # 4. Revert if rejected.
                        
                        # This is effectively what we do for Add/Delete.
                        # For Swap, it's just Add + Delete.
                        # We can just call the logic for Add and Delete sequentially?
                        # No, because the acceptance is for the combined move.
                        
                        # Let's just do the explicit loop.
                        # Revert temp changes
                        for v2 in balls[c]:
                            cov[v2] += 1
                        for v2 in balls[v]:
                            cov[v2] -= 1
                        
                        # Calculate delta_U properly
                        # affected = set(balls[c]) | set(balls[v])
                        # This is slow due to set creation.
                        
                        # Let's use the property that delta_U = (# of v where cov becomes 0) - (# of v where cov was 0 and becomes >0)
                        # We can compute this by iterating over balls[c] and balls[v].
                        
                        # Let's just use the "Apply, Check, Revert" pattern for Swap.
                        # It's O(26) to apply/revert.
                        # To check U, we need to know if any cov became 0 or if any 0 became >0.
                        # We can track this during the application.
                        
                        # Re-apply
                        delta_U = 0
                        # Remove c
                        for v2 in balls[c]:
                            if cov[v2] == 1:
                                delta_U += 1 # Will become 0
                            cov[v2] -= 1
                        # Add v
                        for v2 in balls[v]:
                            if cov[v2] == 0:
                                delta_U -= 1 # Was 0, will become 1
                            cov[v2] += 1
                            
                        # Now cov is updated for the swap.
                        delta_size = 0
                        delta_E = delta_size + M * delta_U
                        
                        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                            # Accept
                            accepted = True
                            in_code[c] = False
                            in_code[v] = True
                            current_code[i] = v
                            U += delta_U
                            E += delta_E
                            
                            if U == 0 and len(current_code) < local_best_size:
                                local_best_size = len(current_code)
                                no_improve_steps = 0
                                if len(current_code) < best_size:
                                    best_size = len(current_code)
                                    best_code = current_code[:]
                            else:
                                no_improve_steps += 1
                        else:
                            # Reject: Revert
                            # Reverse of above
                            # Remove v
                            for v2 in balls[v]:
                                cov[v2] -= 1
                            # Add c
                            for v2 in balls[c]:
                                cov[v2] += 1
                            # U and E unchanged
                            
            # Cool down
            T *= alpha
            
            # Optional: Restart local search if stuck
            if no_improve_steps > 1000:
                break

    # Convert best_code indices to digits
    result = [idx_to_digits(c) for c in best_code]
    return result
