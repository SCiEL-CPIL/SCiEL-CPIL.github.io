    for e in range(1, N + 1):
        partner = inv[e]
        if partner == e:
            # Self-inverse: e cannot be in any valid interval
            s = e + 1
        else:
            # If partner is in the current window [s, e-1], advance s past it
            if s <= partner < e:
                s = partner + 1

        cur_len = e - s + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = s
