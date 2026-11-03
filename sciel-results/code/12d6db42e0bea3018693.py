    while unvisited_jumps:
        if time.time() > deadline:
            break
            
        if current_state_idx in unvisited_jumps:
            # We can take the Jump edge from current_state_idx
            word.append(jump_symbol[current_state_idx])
            next_idx = jump_next[current_state_idx]
            unvisited_jumps.remove(current_state_idx)
            current_state_idx = next_idx
        else:
            # Must take Rotate
            word.append(rotate_symbol[current_state_idx])
            current_state_idx = rotate_next[current_state_idx]
