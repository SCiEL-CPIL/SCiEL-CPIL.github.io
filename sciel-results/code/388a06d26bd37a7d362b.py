    def is_valid_addition_optimized(x, S, u=None):
        s_list = [y for y in S if y != u]
        
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                
                sum_ab = a + b
                in_set_ab = (sum_ab in S and sum_ab != u) or (sum_ab == x)
                if not in_set_ab:
                    continue
                
                sum_ax = x + a
                in_set_ax = (sum_ax in S and sum_ax != u)
                # sum_ax == x is impossible for a > 0
                if not in_set_ax:
                    continue
                
                sum_bx = x + b
                in_set_bx = (sum_bx in S and sum_bx != u)
                if not in_set_bx:
                    continue
                
                return False
        return True
