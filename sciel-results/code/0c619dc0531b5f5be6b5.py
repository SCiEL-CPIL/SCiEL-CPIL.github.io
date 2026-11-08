    for first_size in range(1, size + 1):
        first_group = list(range(first_size))
        for rest in gen_partitions(size - first_size):
            shifted = [[x + first_size for x in g] for g in rest]
            results.append([first_group] + shifted)
