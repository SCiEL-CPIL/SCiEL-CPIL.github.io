import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128

    # Precompute 4-term APs ending at each position (0-indexed)
    # aps_ending[i] = list of (a, d) where a+3d = i, a >= 0
    aps_ending = [[] for _ in range(N)]
    for i in range(N):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))

    # Precompute 3-term APs ending at each position for lookahead
    # threes_ending[i] = list of (a, d) where a+2d = i, a >= 0
    threes_ending = [[] for _ in range(N)]
    for i in range(N):
        for d in range(1, i // 2 + 1):
            a = i - 2 * d
            if a >= 0:
                threes_ending[i].append((a, d))

    def verify(colors):
        """Verify all 4-term APs have >= 3 distinct colors."""
        for i in range(N):
            for a, d in aps_ending[i]:
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2 * d]
                c3 = colors[i]
                # Check distinct count >= 3
                if c0 == c1 == c2 == c3:
                    return False
                # Check not exactly 2 colors with 3+ same
                s = {c0, c1, c2, c3}
                if len(s) == 2:
                    # Count occurrences
                    cnt0 = (c0 == c1) + (c0 == c2) + (c0 == c3)
                    if cnt0 >= 3 or cnt0 <= 0:
                        # Actually if cnt0 >= 3, then c0 appears 3+ times
                        # This means only 2 distinct colors, which is invalid
                        return False
                    cnt1 = (c1 == c2) + (c1 == c3) + (c1 == c0)
                    # If we're here, len(s)==2, so either pattern is 3-1 or 2-2
                    # 2-2 is valid (4 distinct... no, 2 distinct with 2 each is valid)
                    # Wait: 2 distinct colors total, need >=3 distinct. So ANY 2-color AP is invalid.
                    return False
        return True

    def greedy_construct(max_color):
        """Greedy: assign smallest valid color at each position."""
        colors = [None] * N

        for i in range(N):
            if time.time() > deadline:
                return None

            for c in range(max_color):
                valid = True
                # Check 4-term APs ending at i
                for a, d in aps_ending[i]:
                    vals = {colors[a], colors[a + d], colors[a + 2 * d], c}
                    if len(vals) < 3:
                        valid = False
                        break
                if not valid:
                    continue

                # Lookahead: for 3-term APs where i is the 3rd term
                # AP (a, a+d, a+2d, a+3d) where a+2d = i
                # If we assign c to i, then for the 4th term at a+3d,
                # we need that the set {colors[a], colors[a+d], c, ?} can have >=3 colors
                # This means: if colors[a] == colors[a+d] == c, then ? must differ from c
                # If colors[a] == c != colors[a+d], then ? can be anything
                # The critical case: if colors[a] == colors[a+d] == c, then position a+3d
                # must not be c. But we don't know a+3d's color yet.
                # The seed's lookahead: for 3-term AP (a, a+d, a+2d) ending at i,
                # ensure c != colors[a] OR c != colors[a+d]
                # i.e., NOT (c == colors[a] AND c == colors[a+d])
                for a, d in threes_ending[i]:
                    if a + 3 * d < N:
                        if c == colors[a] and c == colors[a + d]:
                            valid = False
                            break

                if valid:
                    colors[i] = c
                    break
            else:
                return None

        return colors

    def solve_csp(k, dl):
        """CSP solver with backtracking and forward checking."""
        colors = [None] * N
        colors[0] = 0

        # Domain for each position: bitmask of valid colors
        full_mask = (1 << k) - 1
        domain = [full_mask] * N
        domain[0] = 1  # Only color 0

        def check_ap_ending(pos, color):
            """Check if assigning 'color' to 'pos' satisfies all 4-term APs ending at pos."""
            for a, d in aps_ending[pos]:
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2 * d]
                if c0 is None or c1 is None or c2 is None:
                    continue
                vals = {c0, c1, c2, color}
                if len(vals) < 3:
                    return False
            return True

        def forward_check(pos, color, new_domain):
            """Update domains for future positions after assigning color to pos."""
            for y in range(pos + 1, N):
                new_mask = full_mask
                for a, d in aps_ending[y]:
                    # AP: (a, a+d, a+2d, y)
                    # If pos is one of a, a+d, a+2d, this constraint is partially determined
                    assigned = []
                    if a == pos:
                        assigned.append(color)
                    elif colors[a] is not None:
                        assigned.append(colors[a])
                    if a + d == pos:
                        assigned.append(color)
                    elif colors[a + d] is not None:
                        assigned.append(colors[a + d])
                    if a + 2 * d == pos:
                        assigned.append(color)
                    elif colors[a + 2 * d] is not None:
                        assigned.append(colors[a + 2 * d])

                    if len(assigned) == 3:
                        # All three previous positions determined; check if any color for y works
                        # We need len({c0, c1, c2, cy}) >= 3 for some cy
                        # This fails only if all valid cy give < 3 distinct
                        # With 3 fixed colors, if they use < 2 distinct, we need cy to add a 3rd
                        # If they use 2 distinct, any cy works (either same or new)
                        # If they use 3 distinct, any cy works
                        # Actually: with 3 fixed values, the min distinct with a 4th is:
                        #   3 fixed all same: need cy != that color -> k-1 choices
                        #   3 fixed with 2 same: any cy works
                        #   3 fixed all different: any cy works
                        # So only fails if k=1 and all 3 same, but k>=3 so always ok
                        pass

                # Simpler: just check if domain[y] can be restricted
                # For each AP ending at y that includes pos, if the other 3 are assigned,
                # check if there exists a valid color for y
                for a, d in aps_ending[y]:
                    c0 = color if a == pos else colors[a]
                    c1 = color if (a + d) == pos else colors[a + d]
                    c2 = color if (a + 2 * d) == pos else colors[a + 2 * d]

                    if c0 is not None and c1 is not None and c2 is not None:
                        # Check if any color in new_mask works for position y
                        found = False
                        mask = new_mask
                        while mask:
                            bit = mask & -mask
                            cy = bit.bit_length() - 1
                            mask -= bit
                            vals = {c0, c1, c2, cy}
                            if len(vals) >= 3:
                                found = True
                                break
                        if not found:
                            return False, new_mask

                new_domain[y] = new_mask

            return True, new_domain

        def backtrack(pos):
            if time.time() > dl:
                return None

            if pos >= N:
                return colors[:]

            # Try each color in domain
            mask = domain[pos]
            while mask:
                bit = mask & -mask
                c = bit.bit_length() - 1
                mask -= bit

                if not check_ap_ending(pos, c):
                    continue

                # Lookahead check
                ok = True
                for a, d in threes_ending[pos]:
                    if a + 3 * d < N:
                        if c == colors[a] and c == colors[a + d]:
                            ok = False
                            break
                if not ok:
                    continue

                colors[pos] = c
                old_domain = domain[pos + 1:N].copy() if pos + 1 < N else []

                fc_ok, new_domain = forward_check(pos, c, domain[pos + 1:N].copy() if pos + 1 < N else [])
                if fc_ok:
                    domain[pos + 1:N] = new_domain
                    res = backtrack(pos + 1)
                    if res is not None:
                        return res

                # Undo
                colors[pos] = None
                if pos + 1 < N:
                    domain[pos + 1:N] = old_domain

            return None

        return backtrack(1)

    # Try k=3 first
    sol = solve_csp(3, start_time + 35.0)
    if sol is not None:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        result = [mapping[c] for c in sol]
        if verify(result):
            return result

    # Try k=4
    sol = solve_csp(4, start_time + 65.0)
    if sol is not None:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        result = [mapping[c] for c in sol]
        if verify(result):
            return result

    # Fallback: greedy with more colors
    for max_c in [5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        if time.time() > deadline - 2:
            break
        sol = greedy_construct(max_c)
        if sol is not None:
            used = sorted(set(sol))
            mapping = {c: i for i, c in enumerate(used)}
            result = [mapping[c] for c in sol]
            if verify(result):
                return result

    # Last resort: seed-style greedy
    colors = []
    for x in range(1, N + 1):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                i = x - 3 * d
                if i < 1:
                    break
                j = i + d
                k = i + 2 * d
                vals = {colors[i - 1], colors[j - 1], colors[k - 1], color}
                if len(vals) < 3:
                    ok = False
                    break
            if not ok:
                continue
            # Lookahead
            for d in range(1, (x - 1) // 2 + 1):
                i = x - 2 * d
                if i < 1:
                    break
                j = i + d
                if x + d <= N and color == colors[i - 1] and color == colors[j - 1]:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
