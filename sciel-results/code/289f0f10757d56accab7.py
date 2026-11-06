        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        # Check distinctness
        is_dup = False
        for k in range(20):
            if k != idx and points[k] == new_p:
                is_dup = True
                break
        if is_dup:
            points[idx] = [old_x, old_y]
            continue
        points[idx] = new_p
