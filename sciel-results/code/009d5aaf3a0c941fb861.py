    def local_to_global_A(l, b):
        i_loc = l // 2
        j_loc = l % 2
        i_glob = i_loc + 2 * (b // 2)
        j_glob = j_loc + 2 * (b % 2)
        return i_glob * 4 + j_glob

    def local_to_global_V(l, b):
        j_loc = l // 2
        k_loc = l % 2
        j_glob = j_loc + 2 * (b // 2)
        k_glob = k_loc + 2 * (b % 2)
        return j_glob * 9 + k_glob # p=9

    def local_to_global_W(l, b):
        k_loc = l // 2
        i_loc = l % 2
        k_glob = k_loc + 2 * (b // 2)
        i_glob = i_loc + 2 * (b % 2)
        return k_glob * 4 + i_glob # n=4
