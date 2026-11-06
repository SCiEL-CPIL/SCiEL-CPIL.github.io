                if m == 0:
                    st = completion[0]
                else:
                    st = completion[m-1]
                    if completion[m] > st:
                        st = completion[m]
