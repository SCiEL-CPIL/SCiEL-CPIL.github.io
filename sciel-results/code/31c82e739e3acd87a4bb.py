def _check_4cycle(voltages):
    """Check no 4-closed-walk in base has voltage 0 mod 40."""
    v = voltages
    for i in range(4):
        vi = v[i]
        for k in range(4):
            if i == k: continue
            vk = v[k]
            for j in range(4):
                vij = vi[j]
                vkj = vk[j]
                for l in range(4):
                    if j == l: continue
                    s = (vij - vkj + vk[l] - vi[l]) % 40
                    if s == 0:
                        return False
    return True
