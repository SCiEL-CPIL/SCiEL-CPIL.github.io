    while time.time() - start_time < time_limit:
        # Try to find any valid insertion
        found = False
        # Try a random subset of unused elements to save time
        unused_list = list(current_state.unused)
        random.shuffle(unused_list)
        
        for x in unused_list:
            positions = list(range(len(current_state.seq) + 1))
            random.shuffle(positions)
            for i in positions:
                if can_insert(current_state, x, i):
                    current_state.insert(x, i)
                    found = True
                    if len(current_state.seq) > best_len:
                        best_len = len(current_state.seq)
                        best_seq = list(current_state.seq)
                    break
            if found:
                break
                
        if not found:
            # Stuck. Delete a random element to change landscape.
            if current_state.seq:
                i = random.randint(0, len(current_state.seq) - 1)
                current_state.delete(i)
