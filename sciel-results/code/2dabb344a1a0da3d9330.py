# Inside Swap
            # ...
            if accept:
                # Check for duplicates
                temp_blocks = [b for b in current_blocks if b != b_old_id]
                if new_b_id in temp_blocks:
                    # Duplicate, reject
                    pass
                else:
                    current_blocks = temp_blocks
                    current_blocks.append(new_b_id)
                    current_counts = new_counts
                    current_cost_val = new_cost_val
                    # ... update best
