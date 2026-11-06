if quick_check(candidate):
    if is_sorted_network(candidate):
        best = candidate
        removed_any = True
    else:
        i += 1
else:
    i += 1
