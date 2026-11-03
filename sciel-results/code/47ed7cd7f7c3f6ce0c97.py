    def add_block(u_offset, v_offset, w_offset, sign=1):
        # ...
        for c in range(7):
            r = c
            # ...
            for k in range(4):
                if W2[k, c] != 0:
                    W4[w_offset + k, r] += sign * W2[k, c]
