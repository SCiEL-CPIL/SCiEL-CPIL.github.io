def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_trans_a = None
    best_trans_b = None
    best_score = -1
    
    def evaluate(trans_a, trans_b):
        return reset_threshold(trans_a, trans_b)
    
    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def cycle_automaton():
        # a is a cycle, b is random
        ta = [(i + 1) % N for i in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def local_search(init_a, init_b, max_steps=100):
        """
        Random Hill Climbing: pick a random neighbor, accept if better.
        This is fast and allows many restarts.
        """
        current_a = list(init_a)
        current_b = list(init_b)
        current_score = evaluate(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        steps = 0
        while steps < max_steps and time.time() - start_time < timeout:
            # Pick a random move
            if random.random() < 0.5:
                # Change a random state in trans_a
                idx = random.randint(0, N - 1)
                old_val = current_a[idx]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_a[idx] = new_val
                new_score = evaluate(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    steps += 1
                else:
                    current_a[idx] = old_val
            else:
                # Change a random state in trans_b
                idx = random.randint(0, N - 1)
                old_val = current_b[idx]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_b[idx] = new_val
                new_score = evaluate(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    steps += 1
                else:
                    current_b[idx] = old_val
            
            # Early stop if we hit the theoretical max (N*(N-1)/2)
            if current_score == N * (N - 1) // 2:
                break

        return current_a, current_b, current_score

    # Main loop: Many random restarts
    # Bias towards cycle-based initialization as they are more likely to be good
    while time.time() - start_time < timeout:
        # 70% cycle-based, 30% fully random
        if random.random() < 0.7:
            init_a, init_b = cycle_automaton()
        else:
            init_a, init_b = random_automaton()
            
        best_a, best_b, score = local_search(init_a, init_b, max_steps=50)
        
        if score > best_score:
            best_score = score
            best_trans_a = best_a
            best_trans_b = best_b

    if best_trans_a is None or best_score < 0:
        best_trans_a = [(i + 1) % N for i in range(N)]
        best_trans_b = [0] * N
        best_score = reset_threshold(best_trans_a, best_trans_b)
    
    return best_trans_a, best_trans_b
