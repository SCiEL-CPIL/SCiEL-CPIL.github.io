import time

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    max_depth = 25
    
    # Two beams: one preferring small last values, one preferring large
    beam_width_small = 100
    beam_width_large = 100
    
    beam_small = [([], set(), 0)]
    beam_large = [([], set(), 0)]
    
    best_sequence = []
    best_len = 0
    
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Expand beam_small
        next_small = []
        for seq, sums, last_val in beam_small:
            start_val = last_val + 1 if last_val > 0 else 1
            for x in range(start_val, 257):
                valid = True
                temp_sum = x
                if temp_sum in sums:
                    valid = False
                else:
                    for a in reversed(seq):
                        temp_sum += a
                        if temp_sum in sums:
                            valid = False
                            break
                if valid:
                    new_sums_list = [x]
                    temp_sum = x
                    for a in reversed(seq):
                        temp_sum += a
                        new_sums_list.append(temp_sum)
                    new_sums_set = sums.union(new_sums_list)
                    new_seq = seq + (x,)
                    next_small.append((new_seq, new_sums_set, x))
        
        if next_small:
            next_small.sort(key=lambda item: item[2]) # Ascending
            beam_small = next_small[:beam_width_small]
            if len(beam_small[0][0]) > best_len:
                best_len = len(beam_small[0][0])
                best_sequence = list(beam_small[0][0])
        else:
            beam_small = []

        # Expand beam_large
        next_large = []
        for seq, sums, last_val in beam_large:
            start_val = last_val + 1 if last_val > 0 else 1
            for x in range(start_val, 257):
                valid = True
                temp_sum = x
                if temp_sum in sums:
                    valid = False
                else:
                    for a in reversed(seq):
                        temp_sum += a
                        if temp_sum in sums:
                            valid = False
                            break
                if valid:
                    new_sums_list = [x]
                    temp_sum = x
                    for a in reversed(seq):
                        temp_sum += a
                        new_sums_list.append(temp_sum)
                    new_sums_set = sums.union(new_sums_list)
                    new_seq = seq + (x,)
                    next_large.append((new_seq, new_sums_set, x))
        
        if next_large:
            next_large.sort(key=lambda item: item[2], reverse=True) # Descending
            beam_large = next_large[:beam_width_large]
            if len(beam_large[0][0]) > best_len:
                best_len = len(beam_large[0][0])
                best_sequence = list(beam_large[0][0])
        else:
            beam_large = []

        if not beam_small and not beam_large:
            break

    return best_sequence
