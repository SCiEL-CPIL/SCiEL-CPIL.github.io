    def reconstruct(weights, target):
        # Meet in the middle for reconstruction
        n_w = len(weights)
        mid_r = n_w // 2
        w_r1 = weights[:mid_r]
        w_r2 = weights[mid_r:]
        
        def gen_with_coeffs(w):
            res = [(0, (0,)*len(w))]
            for i, wi in enumerate(w):
                new_res = []
                for s, c in res:
                    c_list = list(c)
                    c_list[i] = -1
                    new_res.append((s - wi, tuple(c_list)))
                    c_list[i] = 0
                    new_res.append((s, tuple(c_list)))
                    c_list[i] = 1
                    new_res.append((s + wi, tuple(c_list)))
                res = new_res
            return res
