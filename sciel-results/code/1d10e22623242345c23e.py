<<<<<<< SEARCH
    def local_improve(rows, seed, max_iters=200):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        # Maintain the used-set incrementally for speed
        used_set = _compute_all_diffs(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove this element and its diffs from used_set
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - new_val
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope
=======
    def local_improve(rows, seed, max_iters=200):
        """Try to reduce scope by shifting elements left using bitset for speed."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        # Maintain the used-set incrementally for speed
        used_set = _compute_all_diffs(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove this element and its diffs from used_set
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - new_val
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope
>>>>>>> REPLACE

