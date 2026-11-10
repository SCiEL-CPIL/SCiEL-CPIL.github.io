# Inside greedy_hitting_set
unhit_list = list(range(n_faces))
# ...
# Instead of set, use list and a boolean array for unhit status?
# Or just stick to set and accept the list() cost.
# Given n_faces=672, list() is cheap.
