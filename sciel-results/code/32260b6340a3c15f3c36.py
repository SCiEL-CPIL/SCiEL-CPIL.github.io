<<<<<<< SEARCH
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    best_trans = None
    best_len = 0
    
    # Family 1: Pinwheel variants
    # a = cycle, b breaks at position k
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i + 1 for i in range(N)]
        b[N-1] = break_pos
        trans = [a, b]
        try:
            L = _bfs_reset_length(trans)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 2: a = cycle, b = identity except one state maps to 0
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i for i in range(N)]
        b[break_pos] = 0
        trans = [a, b]
        try:
            L = _bfs_reset_length(trans)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 3: a = cycle, b maps i -> i+1 except last maps to some k
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[N-1] = k
        trans = [a, b]
        try:
            L = _bfs_reset_length(trans)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 4: a = shift by 1, b = shift by 1 except one "jump"
    for jump_src in range(N):
        for jump_dst in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [(i + 1) % N for i in range(N)]
            b[jump_src] = jump_dst
            trans = [a, b]
            try:
                L = _bfs_reset_length(trans)
                if L > best_len:
                    best_len = L
                    best_trans = trans
            except:
                pass
            if time.time() > deadline:
                break
    
    # Family 5: a = cycle, b = i -> i+2 mod N (different stride)
    for stride in range(2, N):
        if __import__('math').gcd(stride, N) == 1:
            a = [(i + 1) % N for i in range(N)]
            b = [(i + stride) % N for i in range(N)]
            # This is two permutations, likely not synchronizing
            # Skip unless it has a fixed point structure
            pass
    
    # Family 6: a = cycle, b = constant map to 0
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 7: a = cycle, b = i -> max(i-1, 0) (shift left with clamp)
    a = [(i + 1) % N for i in range(N)]
    b = [max(i - 1, 0) for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 8: a = cycle, b = i -> (i * 2) % N
    a = [(i + 1) % N for i in range(N)]
    b = [(i * 2) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 9: a = cycle, b = i -> (i+1) % N for i < N-2, b(N-2)=0, b(N-1)=N-1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[N-2] = 0
    b[N-1] = N-1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 10: a = cycle, b = i -> i for i even, b = i+1 for i odd
    a = [(i + 1) % N for i in range(N)]
    b = [i if i % 2 == 0 else (i + 1) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 11: a = cycle, b = i -> (i + 1) % N except b(0) = N-1 (reverse one edge)
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[0] = N - 1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 12: a = cycle, b = i -> (i - 1) % N (reverse cycle)
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 13: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=N-2
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(N - 2)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 14: a = cycle, b = i -> (i+1)%N for i < N-3, b(N-3)=0, b(N-2)=N-2, b(N-1)=N-1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[N-3] = 0
    b[N-2] = N-2
    b[N-1] = N-1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 15: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(1)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 16: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=2
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(2)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 17: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=N-3
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(N - 3)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 18: a = cycle, b = i -> (i+1)%N except b(0)=0 (one fixed point)
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[0] = 0
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 19: a = cycle, b = i -> (i+1)%N except b(1)=0
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[1] = 0
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 20: a = cycle, b = i -> (i+1)%N except b(2)=0
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[2] = 0
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    if best_trans is not None:
        return best_trans
    else:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
=======
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    random.seed(42)
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Phase 1: Quick scan of structured families to get a good starting point
    # Family: a = cycle, b = cycle with one state redirected
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[N-1] = k
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
        if time.time() > deadline - 60:
            break
    
    # Family: a = cycle, b = shift with one merge
    for src in range(N):
        for dst in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [(i + 1) % N for i in range(N)]
            b[src] = dst
            L = _bfs_reset_length_bitmask(a, b)
            if L > best_len:
                best_len = L
                best_a = a[:]
                best_b = b[:]
            if time.time() > deadline - 60:
                break
    
    # Phase 2: Simulated annealing to explore non-canonical automata
    if best_a is None:
        # Fallback initial
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = _bfs_reset_length_bitmask(best_a, best_b)
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    T = T_start
    
    while time.time() < deadline - 10:
        # Generate neighbor: mutate one transition in one letter
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_dest = current_a[state] if letter == 0 else current_b[state]
        new_dest = random.randint(0, N - 1)
        
        if letter == 0:
            current_a[state] = new_dest
        else:
            current_b[state] = new_dest
        
        new_len = _bfs_reset_length_bitmask(current_a, current_b)
        
        # Acceptance criterion
        if new_len > current_len:
            # Always accept improvement
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Accept with probability exp(-(current_len - new_len) / T)
            delta = current_len - new_len
            if delta == 0:
                # Same length, accept with some probability to explore
                if random.random() < 0.5:
                    current_len = new_len
                else:
                    # Revert
                    if letter == 0:
                        current_a[state] = old_dest
                    else:
                        current_b[state] = old_dest
            else:
                prob = min(1.0, 0.0001 * (delta / T))
                if random.random() < prob:
                    current_len = new_len
                else:
                    # Revert
                    if letter == 0:
                        current_a[state] = old_dest
                    else:
                        current_b[state] = old_dest
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Phase 3: Local improvement pass
    if time.time() < deadline - 5:
        # Try single-edge mutations from best
        for letter in range(2):
            for state in range(N):
                for new_dest in range(N):
                    if letter == 0:
                        old = best_a[state]
                        best_a[state] = new_dest
                        L = _bfs_reset_length_bitmask(best_a, best_b)
                        best_a[state] = old
                    else:
                        old = best_b[state]
                        best_b[state] = new_dest
                        L = _bfs_reset_length_bitmask(best_a, best_b)
                        best_b[state] = old
                    if L > best_len:
                        best_len = L
                        if letter == 0:
                            best_a[state] = new_dest
                        else:
                            best_b[state] = new_dest
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    return [best_a, best_b]
>>>>>>> REPLACE
