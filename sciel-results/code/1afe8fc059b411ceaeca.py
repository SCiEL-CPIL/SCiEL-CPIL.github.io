    def run_once():
        A = []
        B = []
        sum_bits_A = 0
        diff_bits_A = 0
        sum_bits_B = 0
        diff_bits_B = 0
        
        steps = 0
        max_steps = 1000 # Prevent infinite loops in a single run
        
        while time.time() - start_time < DEADLINE and steps < max_steps:
            steps += 1
            side = random.choice([0, 1])
            added = False
            
            for s in (side, 1 - side):
                if s == 0:
                    if len(A) >= MAX_EACH: continue
                    elems = A
                    sum_bits = sum_bits_A
                    diff_bits = diff_bits_A
                    other_diff_bits = diff_bits_B
                else:
                    if len(B) >= MAX_EACH: continue
                    elems = B
                    sum_bits = sum_bits_B
                    diff_bits = diff_bits_B
                    other_diff_bits = diff_bits_A
                
                elems_set = set(elems)
                candidates = [x for x in all_nums if x not in elems_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
                        elems.append(x)
                        new_sum_bits = sum_bits | (1 << (x + x))
                        new_diff_bits = diff_bits
                        for y in elems[:-1]:
                            new_sum_bits |= (1 << (x + y))
                            d = abs(x - y)
                            new_diff_bits |= (1 << d)
                        
                        if s == 0:
                            sum_bits_A = new_sum_bits
                            diff_bits_A = new_diff_bits
                        else:
                            sum_bits_B = new_sum_bits
                            diff_bits_B = new_diff_bits
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                # Perturbation: Remove a random element to unblock
                if A and (not B or random.random() < 0.5):
                    idx = random.randint(0, len(A) - 1)
                    A.pop(idx)
                    sum_bits_A = get_sum_bits(A)
                    diff_bits_A = get_diff_bits(A)
                elif B:
                    idx = random.randint(0, len(B) - 1)
                    B.pop(idx)
                    sum_bits_B = get_sum_bits(B)
                    diff_bits_B = get_diff_bits(B)
                else:
                    break # Both empty, restart

        return A, B
