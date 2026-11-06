best_num = initial_num  # integer numerator of best sum
best_idx = initial_indices
nodes = 0
deadline = time.time() + 80  # safety margin

def dfs(i, num, chosen):
    nonlocal best_num, best_idx, nodes
    # bound
    if num + suffix[i] <= best_num:
        return
    if i == 47:
        return
    nodes += 1
    if nodes % 10000 == 0 and time.time() > deadline:
        raise Timeout
    # take item i (if valid)
    new_num = num + vals[i]
    if new_num < L:  # strictly below 1
        chosen.append(i)
        dfs(i+1, new_num, chosen)
        chosen.pop()
    # skip item i
    dfs(i+1, num, chosen)
