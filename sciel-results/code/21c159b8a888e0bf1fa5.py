import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    target_diffs = set(range(1, 31))

    def _check_solution(elements):
        """Verify Sidon property and coverage of all differences 1..30."""
        if 0 not in elements:
            return False, float('inf')
        elements = sorted(elements)
        if len(elements) > 32:
            return False, float('inf')
        diffs = set()
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False, float('inf')
                diffs.add(d)
        if not target_diffs.issubset(diffs):
            return False, float('inf')
        return True, elements[-1]

    def _dfs(elements, seen_diffs, covered_diffs, depth, best_max_elem):
        """Difference-driven DFS: pick uncovered diff, enumerate valid coverings."""
        if time.time() > deadline:
            return None

        if not covered_diffs:
            # All differences covered
            return list(elements)

        # Pick the "hardest" uncovered difference (most constrained)
        # Heuristic: prefer differences that have fewer valid coverings
        uncovered = sorted(covered_diffs)

        # For each uncovered difference, find valid (e, x) pairs
        # where x = e + d, x not in elements, and all new diffs are valid
        best_solution = None
        best_max = best_max_elem

        # Try covering differences in order of "difficulty"
        # A difference d is harder if fewer valid placements exist
        diff_order = []
        for d in uncovered:
            count = 0
            for e in elements:
                x = e + d
                if x in elements:
                    continue
                # Check if x creates duplicate differences
                valid = True
                for e2 in elements:
                    if e2 == e:
                        continue
                    diff = abs(x - e2)
                    if diff in seen_diffs:
                        valid = False
                        break
                if valid:
                    count += 1
            diff_order.append((count, d))

        diff_order.sort()  # Try most constrained first

        for _, d in diff_order:
            # Enumerate valid placements for difference d
            placements = []
            for e in elements:
                x = e + d
                if x in elements:
                    continue
                # Check Sidon constraint
                valid = True
                new_diffs = set()
                for e2 in elements:
                    if e2 == e:
                        continue
                    diff = abs(x - e2)
                    if diff in seen_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if not valid:
                    continue
                # Also check that x - e = d is not already seen
                # (it shouldn't be since d is uncovered)
                placements.append((x, new_diffs))

            if not placements:
                # This difference cannot be covered with current elements
                # We need to add new elements first - but in this formulation,
                # we only add elements that directly cover a difference.
                # If no placement exists, this branch is dead unless we can
                # add elements that don't directly cover d but enable future coverage.
                # For now, skip this difference (it might be coverable later
                # after adding more elements).
                continue

            # Sort placements: prefer smaller x
            placements.sort(key=lambda p: p[0])

            for x, new_diffs in placements:
                if time.time() > deadline:
                    return best_solution

                new_elements = elements + [x]
                new_seen = seen_diffs | new_diffs
                new_covered = covered_diffs - {d}
                # Remove any other differences that are now covered
                new_covered -= new_diffs

                # Pruning: check if remaining can still be covered
                if new_covered:
                    slots_left = 32 - len(new_elements)
                    if slots_left <= 0:
                        continue
                    # Upper bound on new differences we can add
                    max_new = 0
                    ne = len(new_elements)
                    for _ in range(slots_left):
                        max_new += ne
                        ne += 1
                    if len(new_covered) > max_new:
                        continue

                # Bound on max element
                if x >= best_max:
                    continue

                result = _dfs(new_elements, new_seen, new_covered, depth + 1, best_max)
                if result is not None:
                    if result[-1] < best_max:
                        best_max = result[-1]
                        best_solution = result

        return best_solution

    # Phase 1: Difference-driven DFS
    # Start with [0], no seen diffs, all 30 diffs uncovered
    solution = _dfs([0], set(), set(target_diffs), 0, float('inf'))

    if solution is not None:
        valid, max_elem = _check_solution(solution)
        if valid:
            return solution

    # Phase 2: Greedy fallback with backtracking
    def _greedy_construct():
        elements = [0]
        seen_diffs = set()
        covered = set(target_diffs)

        while covered:
            if time.time() > deadline:
                break

            # Find the most constrained uncovered difference
            best_d = None
            best_placements = None
            best_count = float('inf')

            for d in sorted(covered):
                placements = []
                for e in elements:
                    x = e + d
                    if x in elements:
                        continue
                    valid = True
                    new_diffs = set()
                    for e2 in elements:
                        if e2 == e:
                            continue
                        diff = abs(x - e2)
                        if diff in seen_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if not valid:
                        continue
                    placements.append((x, new_diffs))

                if len(placements) < best_count:
                    best_count = len(placements)
                    best_d = d
                    best_placements = placements

                if best_count == 1:
                    break

            if best_placements is None or not best_placements:
                # No valid placement for any difference - try adding a new element
                # that doesn't directly cover a difference but enables future coverage
                found = False
                for trial_x in range(elements[-1] + 1, elements[-1] + 200):
                    valid = True
                    new_diffs = set()
                    for e2 in elements:
                        diff = trial_x - e2
                        if diff > 0:
                            if diff in seen_diffs:
                                valid = False
                                break
                            new_diffs.add(diff)
                    if not valid:
                        continue
                    # Check that this doesn't waste too much
                    elements.append(trial_x)
                    seen_diffs.update(new_diffs)
                    covered -= new_diffs
                    found = True
                    break

                if not found:
                    break
                continue

            # Pick the smallest x from the best placements
            best_placements.sort(key=lambda p: p[0])
            x, new_diffs = best_placements[0]
            elements.append(x)
            elements.sort()
            seen_diffs.update(new_diffs)
            covered -= new_diffs
            covered -= {best_d}

        return elements

    result = _greedy_construct()
    valid, max_elem = _check_solution(result)
    if valid:
        return result

    # Phase 3: Last resort - construct a known valid set
    # A Sidon set that covers 1..30
    # Use a simple construction: {0, 1, 4, 10, 18, 28, 38, 49, 61}
    # Check if this works
    fallback = [0, 1, 4, 10, 18, 28, 38, 49, 61]
    valid, max_elem = _check_solution(fallback)
    if valid:
        return fallback

    # Extend fallback if needed
    elements = list(fallback)
    seen_diffs = set()
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            seen_diffs.add(elements[j] - elements[i])

    covered = target_diffs - seen_diffs
    while covered:
        found = False
        for d in sorted(covered):
            for e in elements:
                x = e + d
                if x in elements:
                    continue
                valid = True
                new_diffs = set()
                for e2 in elements:
                    if e2 == e:
                        continue
                    diff = abs(x - e2)
                    if diff in seen_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if not valid:
                    continue
                elements.append(x)
                elements.sort()
                seen_diffs.update(new_diffs)
                covered -= new_diffs
                found = True
                break
            if found:
                break

        if not found:
            # Add a new element
            for trial_x in range(elements[-1] + 1, elements[-1] + 500):
                valid = True
                new_diffs = set()
                for e2 in elements:
                    diff = trial_x - e2
                    if diff > 0:
                        if diff in seen_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                if not valid:
                    continue
                elements.append(trial_x)
                elements.sort()
                seen_diffs.update(new_diffs)
                covered -= new_diffs
                found = True
                break

        if not found:
            break

    return elements

