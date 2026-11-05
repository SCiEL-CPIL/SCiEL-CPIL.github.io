            if len(active_wires) == 0:
                outputs.append(n)
            elif len(active_wires) == 1:
                outputs.append(active_wires[0])
            else:
                # Balanced XOR tree
                wires = active_wires[:]
                while len(wires) > 1:
                    new_wires = []
                    for i in range(0, len(wires) - 1, 2):
                        new_wires.append(create_xor_gate(wires[i], wires[i+1]))
                    if len(wires) % 2 != 0:
                        new_wires.append(wires[-1])
                    wires = new_wires
                outputs.append(wires[0])
