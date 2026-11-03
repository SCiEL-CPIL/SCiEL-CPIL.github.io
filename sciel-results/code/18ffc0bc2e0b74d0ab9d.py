counts[neighbors_arr[c_remove, :]] -= 1  # This is wrong - c_remove was already removed
counts[neighbors_arr[best_w, :]] += 1
